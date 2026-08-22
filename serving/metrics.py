"""What this service tells a monitoring system about itself.

Four things are exported here, and the fourth is the one that makes this
*model* monitoring rather than ordinary service monitoring:

**Request rate** - ``ryvion_serving_requests_total``, a counter labelled by
method, route and status code. Rate is a derivative, so a counter is the right
instrument: Prometheus stores the total and ``rate()`` turns it into requests
per second at whatever window the question needs.

**Error rate** - the same counter. An error rate is not a separate measurement,
it is a ratio of one selection of that counter to all of it, which is why there
is no ``errors_total`` here. A second counter would be a second thing to keep
consistent and would still not answer "what fraction".

**Latency distribution** - ``ryvion_serving_request_duration_seconds``, a
histogram. A mean latency is close to useless: it hides the tail, and the tail
is what a user notices. A histogram keeps bucket counts so that
``histogram_quantile()`` can answer for the 95th percentile, and buckets
aggregate correctly across replicas in a way that a pre-computed percentile
never can.

**Prediction distribution** - ``ryvion_serving_prediction_mpg``, a histogram of
the values the model actually predicted. This is the one an ordinary web
service has no equivalent of, and it is the one the syllabus cares about. A
model that has gone wrong is very often still fast, still returning 200, and
still perfectly healthy by every signal above - what changed is *what it says*.
Watching the shape of its output is how that becomes visible without waiting
for ground truth, which in this domain arrives weeks later or never.

The buckets span roughly the range of the training labels (about 9 to 47 mpg),
so the histogram has resolution where the data lives rather than spending it
all on one bucket.

Two supporting series come with them: ``ryvion_serving_predictions_total``
counts individual records scored, so that "predictions per second" is not
confused with "requests per second" when one request carries a batch, and
``ryvion_serving_model_loaded`` is 1 or 0, which is the readiness endpoint's
answer in a form a dashboard and an alert rule can read.

Cardinality is the thing to be careful with here, and the care taken is in the
``path`` label. It carries the *route template* FastAPI matched, never the raw
URL: a service that labelled by raw path would mint a new time series for every
distinct URL anyone ever requested, so a scanner probing for ``/wp-admin`` a
thousand times would be a thousand series and the cost of that lands on the
monitoring system rather than here. Anything unmatched is labelled
``unmatched`` and stays one series.
"""

import time
from collections.abc import Iterable

from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest
from starlette.requests import Request
from starlette.responses import Response

#: The path this module's own endpoint is served on.
METRICS_PATH = "/metrics"

#: Label used for requests that matched no route, so that 404 probing cannot
#: create unbounded series.
UNMATCHED = "unmatched"

REQUESTS = Counter(
    "ryvion_serving_requests",
    "HTTP requests handled, by method, matched route and status code.",
    labelnames=("method", "path", "status"),
)

LATENCY = Histogram(
    "ryvion_serving_request_duration_seconds",
    "Wall-clock time to handle a request, by method and matched route.",
    labelnames=("method", "path"),
    # A prediction from this model is single-digit milliseconds when it is well
    # and hundreds when it is not, so the buckets are dense where the service
    # should be and thin out afterwards.
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0),
)

PREDICTIONS = Counter(
    "ryvion_serving_predictions",
    "Individual records scored. One request may carry many.",
)

PREDICTION_MPG = Histogram(
    "ryvion_serving_prediction_mpg",
    "Distribution of the miles-per-gallon values the model predicted.",
    # The training labels run from about 9 to 47 mpg. Buckets outside that range
    # exist so that a model predicting somewhere it never saw is visible rather
    # than clipped into an edge bucket.
    buckets=(5, 10, 12.5, 15, 17.5, 20, 22.5, 25, 27.5, 30, 35, 40, 50, 75),
)

MODEL_LOADED = Gauge(
    "ryvion_serving_model_loaded",
    "1 when a model is loaded and this replica can serve, 0 otherwise.",
)

# Set now rather than only in the lifespan hook. An unset gauge simply does not
# appear in the exposition, and an alert on a series that does not exist has
# nothing to fire on - so "no model" would look exactly like "no service".
MODEL_LOADED.set(0)


def observe_predictions(values: Iterable[float]) -> None:
    """Record what the model just said, one observation per scored record."""
    for value in values:
        PREDICTIONS.inc()
        PREDICTION_MPG.observe(value)


def route_label(request: Request) -> str:
    """The matched route template, or ``unmatched``.

    ``request.scope["route"]`` is populated by Starlette's router *after* the
    route is resolved, which is downstream of the middleware that reads it - so
    this is read on the way out of a request, never on the way in.
    """
    route = request.scope.get("route")
    return getattr(route, "path", None) or UNMATCHED


async def record_request(request: Request, call_next):
    """Time one request and count it, whatever it did.

    Registered as HTTP middleware. An unhandled exception is counted as a 500
    and re-raised rather than swallowed: a metric that quietly turned a crash
    into a missing observation would make the error rate look better the worse
    things were going.
    """
    started = time.perf_counter()
    try:
        response = await call_next(request)
    except Exception:
        _observe(request, time.perf_counter() - started, "500")
        raise

    _observe(request, time.perf_counter() - started, str(response.status_code))
    return response


def _observe(request: Request, elapsed: float, status: str) -> None:
    path = route_label(request)
    LATENCY.labels(request.method, path).observe(elapsed)
    REQUESTS.labels(request.method, path, status).inc()


def exposition() -> Response:
    """The current values, in the text format Prometheus scrapes."""
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)
