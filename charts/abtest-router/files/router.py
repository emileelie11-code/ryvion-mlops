"""The traffic router: canary splitting and traffic mirroring, without a mesh.

This is the piece the inherited ``abtest-istio`` chart delegated to Istio. It
does three things and nothing else:

**Split** a percentage of requests between two model versions (canary).
**Pin** a request to a named version with an ``x-api-version`` header, which is
what ``charts/load_test.sh`` has always sent.
**Mirror** every request to a second version, record what that version predicted,
and throw the answer away (shadow).

Why this exists rather than a ``VirtualService``
------------------------------------------------

A service mesh does all of the above and a great deal more - mutual TLS, retries,
circuit breaking, per-hop telemetry - by putting a sidecar proxy next to every
pod and a control plane above them. That is the right answer in production and
the wrong answer on a student laptop: measured on the cluster this repository
builds, Istio costs roughly as much resident memory as the entire rest of the
course. The README carries the numbers.

So the mesh's *shape* is kept and its *weight* is not. This process sits in front
of the two model Services as an ordinary Deployment, speaks plain HTTP, and holds
its state in memory. It is about 20 MB resident. What it gives up is written down
in the README under "What the router is not".

The one rule
------------

**A mirror target is never routable.** The set of upstreams that may answer a
caller and the mirror target are two different things in this file, and the only
way to reach the mirror is to look at what was recorded. Asking for it by name
gets a 409 rather than a prediction. That rule is what makes a shadow deployment
a shadow deployment rather than a second-rate canary, and it is the reason
``serve_targets`` and ``mirror`` are separate attributes below instead of one
list with a flag on it.

Configuration is environment variables, all of them written by the Helm chart:

``ROUTER_STRATEGY``
    ``bluegreen``, ``canary`` or ``shadow``. Reported by ``/_router/status``;
    the routing behaviour follows from the two variables below, not from this.
``ROUTER_ROUTES``
    JSON list of ``{"name", "url", "weight"}`` - the versions that may answer.
``ROUTER_MIRROR``
    JSON ``{"name", "url", "percentage"}``, or empty for no mirroring.
``ROUTER_MIRROR_PATHS``
    Comma-separated paths to mirror. Default ``/predict``.
``ROUTER_PORT``, ``ROUTER_TIMEOUT``, ``ROUTER_RECORD_CAPACITY``
    Listen port, upstream timeout in seconds, and how many mirrored predictions
    to keep for ``/_router/shadow``.

Standard library only, on purpose: the chart mounts this file straight into a
stock ``python:slim`` container from a ConfigMap. There is no image to build, so
there is nothing to rebuild between labs.
"""

import json
import os
import sys
import time
import urllib.error
import urllib.request
from collections import deque
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Lock
from typing import Any

#: Headers that belong to one hop and must not be copied to the next one.
HOP_BY_HOP = frozenset(
    {
        "connection",
        "keep-alive",
        "proxy-authenticate",
        "proxy-authorization",
        "te",
        "trailer",
        "transfer-encoding",
        "upgrade",
        "host",
        "content-length",
    }
)

#: Also dropped when relaying an upstream's answer back: this process stamps its
#: own ``Date`` and ``Server`` on the way out, and two of each confuses clients.
SKIP_ON_RELAY = HOP_BY_HOP | {"date", "server"}

#: Ask for one version by name. The inherited load generator sends this.
PIN_HEADER = "x-api-version"

#: Told back to the caller on every proxied response, so that counting which
#: version answered needs nothing but ``curl -i``.
VERSION_HEADER = "X-Model-Version"

#: Everything under here is answered by the router itself and never proxied.
ROUTER_PREFIX = "/_router"


@dataclass(frozen=True)
class Upstream:
    """One model version the router can reach."""

    name: str
    url: str
    weight: int = 100

    def endpoint(self, path: str) -> str:
        return f"{self.url.rstrip('/')}{path}"


class Splitter:
    """Smooth weighted round-robin - the algorithm nginx uses for upstream weights.

    Each upstream accumulates its own weight on every request; the largest
    accumulator wins and then pays the total back. Over any run of
    ``sum(weights)`` requests each upstream is chosen exactly ``weight`` times,
    and the choices are interleaved rather than arriving in blocks.

    Deterministic, which is the point for a teaching lab: 200 requests at a 10%
    split land 20 on the candidate, not "about 20". A mesh usually splits on a
    hash of the request instead, so its numbers wobble; that difference is worth
    saying out loud when the counts come out suspiciously round.
    """

    def __init__(self, upstreams: list[Upstream]) -> None:
        if not upstreams:
            raise ValueError("a router needs at least one upstream to route to")
        self._upstreams = upstreams
        self._total = sum(max(u.weight, 0) for u in upstreams)
        self._current = {u.name: 0 for u in upstreams}
        self._lock = Lock()

    def pick(self) -> Upstream:
        """The upstream whose turn it is."""
        if self._total <= 0:
            return self._upstreams[0]
        with self._lock:
            for upstream in self._upstreams:
                self._current[upstream.name] += max(upstream.weight, 0)
            chosen = max(self._upstreams, key=lambda u: self._current[u.name])
            self._current[chosen.name] -= self._total
            return chosen


class Sampler:
    """Mirrors ``percentage`` of requests, exactly and without a random number.

    ``percentage`` points accumulate per request and a mirror costs 100 of them,
    so 100 means every request, 50 means every second one, and 0 means never.
    """

    def __init__(self, percentage: int) -> None:
        self._percentage = max(0, min(100, percentage))
        self._credit = 0
        self._lock = Lock()

    def should_sample(self) -> bool:
        if self._percentage <= 0:
            return False
        with self._lock:
            self._credit += self._percentage
            if self._credit >= 100:
                self._credit -= 100
                return True
            return False


class Recorder:
    """The shadow's notebook: what the mirrored version predicted, and never served.

    Bounded on purpose. This is a demonstration of *recording* mirrored
    predictions, and an unbounded list in a long-running process is a memory leak
    with a lesson attached. A real shadow deployment writes these to the same
    place its metrics go; ``/_router/shadow`` is where this one puts them.
    """

    def __init__(self, capacity: int) -> None:
        self._entries: deque[dict[str, Any]] = deque(maxlen=max(1, capacity))
        self._lock = Lock()

    def record(self, entry: dict[str, Any]) -> None:
        with self._lock:
            self._entries.append(entry)

    def entries(self) -> list[dict[str, Any]]:
        with self._lock:
            return list(self._entries)

    def __len__(self) -> int:
        with self._lock:
            return len(self._entries)


class Counters:
    """How many requests each version answered, and how many were mirrored."""

    def __init__(self) -> None:
        self._served: dict[str, int] = {}
        self._mirrored: dict[str, int] = {}
        self._lock = Lock()

    def served(self, name: str) -> None:
        with self._lock:
            self._served[name] = self._served.get(name, 0) + 1

    def mirrored(self, name: str) -> None:
        with self._lock:
            self._mirrored[name] = self._mirrored.get(name, 0) + 1

    def snapshot(self) -> dict[str, dict[str, int]]:
        with self._lock:
            return {"served": dict(self._served), "mirrored": dict(self._mirrored)}


@dataclass(frozen=True)
class Response:
    """What came back from an upstream."""

    status: int
    headers: list[tuple[str, str]]
    body: bytes


def call_upstream(
    url: str, method: str, headers: dict[str, str], body: bytes | None, timeout: float
) -> Response:
    """One HTTP call, with a non-2xx answered rather than raised.

    An upstream's 422 is a real answer and the caller is entitled to see it, so
    ``HTTPError`` - which is also a response object - is relayed as one.
    """
    request = urllib.request.Request(url=url, data=body, headers=headers, method=method)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as answer:
            return Response(answer.status, list(answer.headers.items()), answer.read())
    except urllib.error.HTTPError as failure:
        return Response(failure.code, list(failure.headers.items()), failure.read())


def predictions_in(body: bytes) -> Any:
    """The ``predictions`` field of a serving response, or ``None`` if it has none."""
    try:
        payload = json.loads(body.decode("utf-8"))
    except (ValueError, UnicodeDecodeError):
        return None
    return payload.get("predictions") if isinstance(payload, dict) else None


class Router:
    """The routing decision, the mirroring decision, and the record of both."""

    def __init__(
        self,
        strategy: str,
        serve_targets: list[Upstream],
        mirror: Upstream | None,
        mirror_paths: tuple[str, ...],
        mirror_percentage: int,
        timeout: float,
        capacity: int,
    ) -> None:
        self.strategy = strategy
        self.serve_targets = serve_targets
        self.mirror = mirror
        self.mirror_paths = mirror_paths
        self.mirror_percentage = mirror_percentage
        self.timeout = timeout
        self.splitter = Splitter(serve_targets)
        self.sampler = Sampler(mirror_percentage)
        self.recorder = Recorder(capacity)
        self.counters = Counters()

    def routable(self, name: str) -> Upstream | None:
        return next((u for u in self.serve_targets if u.name == name), None)

    def should_mirror(self, path: str) -> bool:
        if self.mirror is None or path not in self.mirror_paths:
            return False
        return self.sampler.should_sample()

    def status(self) -> dict[str, Any]:
        counts = self.counters.snapshot()
        return {
            "strategy": self.strategy,
            "serving": [
                {"name": u.name, "url": u.url, "weight": u.weight} for u in self.serve_targets
            ],
            "mirroring": (
                None
                if self.mirror is None
                else {
                    "name": self.mirror.name,
                    "url": self.mirror.url,
                    "percentage": self.mirror_percentage,
                    "paths": list(self.mirror_paths),
                    "recorded": len(self.recorder),
                }
            ),
            "requests": counts,
        }


def upstreams_from(raw: str) -> list[Upstream]:
    """Parse ``ROUTER_ROUTES``."""
    return [
        Upstream(name=item["name"], url=item["url"], weight=int(item.get("weight", 100)))
        for item in json.loads(raw or "[]")
    ]


def router_from_environment() -> Router:
    """Build the router the chart's environment variables describe."""
    serve_targets = upstreams_from(os.environ.get("ROUTER_ROUTES", ""))
    raw_mirror = os.environ.get("ROUTER_MIRROR", "").strip()
    mirror = None
    percentage = 0
    if raw_mirror:
        described = json.loads(raw_mirror)
        mirror = Upstream(name=described["name"], url=described["url"])
        percentage = int(described.get("percentage", 100))
    paths = tuple(
        path.strip()
        for path in os.environ.get("ROUTER_MIRROR_PATHS", "/predict").split(",")
        if path.strip()
    )
    return Router(
        strategy=os.environ.get("ROUTER_STRATEGY", "bluegreen"),
        serve_targets=serve_targets,
        mirror=mirror,
        mirror_paths=paths,
        mirror_percentage=percentage,
        timeout=float(os.environ.get("ROUTER_TIMEOUT", "10")),
        capacity=int(os.environ.get("ROUTER_RECORD_CAPACITY", "200")),
    )


def log(message: str, **fields: Any) -> None:
    """One JSON object per line on stdout, which is where a pod's logs live."""
    record = {"ts": time.time(), "msg": message}
    record.update(fields)
    print(json.dumps(record), flush=True)


class Handler(BaseHTTPRequestHandler):
    """Answers the router's own endpoints; proxies everything else."""

    protocol_version = "HTTP/1.1"
    server_version = "abtest-router"
    sys_version = ""

    router: Router

    def do_GET(self) -> None:  # noqa: N802 - the name BaseHTTPRequestHandler dispatches to
        self.handle_any("GET")

    def do_POST(self) -> None:  # noqa: N802 - as above
        self.handle_any("POST")

    def log_message(self, fmt: str, *args: Any) -> None:
        """Silence the default stderr access log; this class logs its own JSON."""

    # --- the router's own surface ------------------------------------------

    def handle_any(self, method: str) -> None:
        path = self.path.split("?", 1)[0]
        if path.startswith(ROUTER_PREFIX):
            self.answer_about_itself(path)
            return
        self.proxy(method, path)

    def answer_about_itself(self, path: str) -> None:
        if path == f"{ROUTER_PREFIX}/healthz":
            self.send_json(200, {"status": "ok"})
        elif path == f"{ROUTER_PREFIX}/status":
            self.send_json(200, self.router.status())
        elif path == f"{ROUTER_PREFIX}/shadow":
            self.send_json(
                200,
                {
                    "mirroring": None if self.router.mirror is None else self.router.mirror.name,
                    "recorded": self.router.recorder.entries(),
                },
            )
        else:
            self.send_json(404, {"detail": f"no router endpoint at {path}"})

    # --- proxying -----------------------------------------------------------

    def choose(self) -> Upstream | None:
        """The version that will answer, honouring an explicit pin if there is one.

        A pin naming the mirror target is refused. That refusal is the shadow
        invariant made testable: there is no request a caller can construct that
        returns the shadow's prediction.
        """
        pinned = self.headers.get(PIN_HEADER)
        if not pinned:
            return self.router.splitter.pick()

        if self.router.mirror is not None and pinned == self.router.mirror.name:
            self.send_json(
                409,
                {
                    "detail": (
                        f"{pinned!r} is a shadow deployment: it receives mirrored traffic "
                        "and its predictions are recorded, never served. "
                        f"See GET {ROUTER_PREFIX}/shadow."
                    )
                },
            )
            return None

        upstream = self.router.routable(pinned)
        if upstream is None:
            known = [u.name for u in self.router.serve_targets]
            self.send_json(404, {"detail": f"no version {pinned!r} here; serving {known}"})
            return None
        return upstream

    def proxy(self, method: str, path: str) -> None:
        body = self.read_body()
        upstream = self.choose()
        if upstream is None:
            return

        forwarded = self.forwardable_headers()
        answer = call_upstream(
            upstream.endpoint(self.path), method, forwarded, body, self.router.timeout
        )
        self.router.counters.served(upstream.name)
        self.relay(answer, upstream.name)

        # After the caller has been answered, never before: mirroring must not
        # be able to slow down or break the request it is copying.
        if self.router.should_mirror(path):
            self.mirror_request(method, forwarded, body, answer, upstream.name)

    def mirror_request(
        self,
        method: str,
        headers: dict[str, str],
        body: bytes | None,
        served: Response,
        served_by: str,
    ) -> None:
        """Send the copy, keep what it predicted, and discard its response."""
        mirror = self.router.mirror
        assert mirror is not None  # noqa: S101 - guarded by should_mirror
        try:
            shadowed = call_upstream(
                mirror.endpoint(self.path), method, headers, body, self.router.timeout
            )
        except OSError as failure:
            log("mirror failed", shadow=mirror.name, error=str(failure))
            return

        entry = {
            "at": time.time(),
            "path": self.path,
            "served_by": served_by,
            "served_predictions": predictions_in(served.body),
            "shadow": mirror.name,
            "shadow_status": shadowed.status,
            "shadow_predictions": predictions_in(shadowed.body),
        }
        self.router.counters.mirrored(mirror.name)
        self.router.recorder.record(entry)
        log("mirrored", **entry)
        # `shadowed` goes out of scope here. That is the whole of the shadow
        # contract: nothing above this line wrote it to the client socket.

    # --- plumbing -----------------------------------------------------------

    def read_body(self) -> bytes | None:
        length = int(self.headers.get("Content-Length") or 0)
        return self.rfile.read(length) if length else None

    def forwardable_headers(self) -> dict[str, str]:
        return {
            name: value
            for name, value in self.headers.items()
            if name.lower() not in HOP_BY_HOP and name.lower() != PIN_HEADER
        }

    def relay(self, answer: Response, version: str) -> None:
        self.send_response(answer.status)
        for name, value in answer.headers:
            if name.lower() not in SKIP_ON_RELAY:
                self.send_header(name, value)
        self.send_header(VERSION_HEADER, version)
        self.send_header("Content-Length", str(len(answer.body)))
        self.end_headers()
        self.wfile.write(answer.body)

    def send_json(self, status: int, payload: dict[str, Any]) -> None:
        encoded = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)


def main() -> int:
    """Serve until killed."""
    router = router_from_environment()
    Handler.router = router
    port = int(os.environ.get("ROUTER_PORT", "8080"))
    log("router starting", port=port, **router.status())
    ThreadingHTTPServer(("0.0.0.0", port), Handler).serve_forever()
    return 0


if __name__ == "__main__":
    sys.exit(main())
