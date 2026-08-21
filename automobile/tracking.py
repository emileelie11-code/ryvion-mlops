"""The experiment-tracking adapter: where a run is recorded, and nothing else.

This is the module that keeps the training code backend-agnostic, and it does so
by being almost empty. It reads the tracking destination from the environment
and names no provider:

``MLFLOW_TRACKING_URI``
    unset - the default - means a local file store under ``./mlruns``. No
    account, no credentials, no network. This is what a student's laptop uses.
    Set, it means whatever it points at: a tracking server, or a managed
    workspace once the package that teaches MLflow to speak to one is installed.
``MLFLOW_EXPERIMENT_NAME``
    the experiment runs are grouped under, when the caller does not pass one.

There is deliberately no ``if azure:`` branch anywhere in this repository, and no
cloud SDK is imported by any module in the local half of it. That is the whole
demonstration: pointing identical training code at a managed backend is a change
of setting, not a change of code. A conditional here would quietly turn that
lesson into a lie.

One consequence of that rule is worth spelling out, because it is where the
default bites. **A model registry needs a database behind it.** The plain
``./mlruns`` directory is enough to record runs, and it is all the training step
needs; the registry the evaluate and register steps use wants somewhere it can
hold a version counter. Locally that is one line of configuration and no
account:

.. code-block:: console

   export MLFLOW_TRACKING_URI=sqlite:///mlflow.db

That destination is still read from the environment like any other, and this
module still names no provider. The README documents it as the local setting;
on the managed backend the same variable points at the workspace instead.

The same rule has a second consequence, and it is the reason :class:`RegistrySurvey`
exists. Because the destination lives in the environment, it is read once per
process and does not follow an operator into a second terminal - so a step can
consult a perfectly valid, perfectly empty store and report "nothing registered
yet" in the same words it would use for a project's genuine first model. The
survey answers "which destination, holding how many models" so the two read
differently. It names no provider either; it reports the URI it was given.

Everything below is a thin adapter over an I/O boundary, so most of it is not
unit tested; the pure modules it serves are. The exception is the survey's
formatting, which is a decision about what an operator gets told and is tested
as one. This module exists so that the entrypoints stay argparse shells rather
than growing tracking calls of their own.
"""

import os
from typing import NamedTuple

import mlflow
from mlflow.tracking import MlflowClient

#: Used when neither the caller nor the environment names an experiment.
DEFAULT_EXPERIMENT_NAME = "automobile-mpg"

#: Used when the caller does not name a registered model. The same name as the
#: experiment, on purpose: one project, one story, two views of it.
DEFAULT_REGISTERED_MODEL_NAME = "automobile-mpg"

ENV_TRACKING_URI = "MLFLOW_TRACKING_URI"
ENV_EXPERIMENT_NAME = "MLFLOW_EXPERIMENT_NAME"

#: The verdict the evaluate step writes onto the candidate run, and the register
#: step refuses to proceed without. Recording it on the run rather than in a file
#: is what lets the two steps run on different machines, which is exactly what
#: happens once they are separate jobs in a pipeline.
GATE_DECISION_TAG = "automobile.gate.decision"
GATE_REASON_TAG = "automobile.gate.reason"
GATE_METRIC_TAG = "automobile.gate.metric"

#: The two values :data:`GATE_DECISION_TAG` ever holds.
PROMOTED = "promote"
REJECTED = "reject"


class TrackingDestination(NamedTuple):
    """Where this run will be recorded, resolved, for printing to the log."""

    uri: str
    experiment_name: str

    @property
    def is_local_file_store(self) -> bool:
        return self.uri.startswith("file:") or "://" not in self.uri


def resolve_experiment_name(experiment_name: str | None = None) -> str:
    """The caller's choice, else the environment's, else the default."""
    return experiment_name or os.environ.get(ENV_EXPERIMENT_NAME) or DEFAULT_EXPERIMENT_NAME


def configure_tracking(experiment_name: str | None = None) -> TrackingDestination:
    """Select the experiment and report where runs are going to land.

    The tracking URI itself is never set here: MLflow reads
    ``MLFLOW_TRACKING_URI`` from the environment on its own, and letting it do
    that is precisely what leaves the destination in the operator's hands rather
    than in this repository's source.
    """
    resolved = resolve_experiment_name(experiment_name)
    mlflow.set_experiment(resolved)
    return TrackingDestination(uri=mlflow.get_tracking_uri(), experiment_name=resolved)


class Incumbent(NamedTuple):
    """The registered model version a candidate is measured against."""

    version: int
    run_id: str
    metrics: dict[str, float]


def run_metrics(run_id: str) -> dict[str, float]:
    """Every metric recorded on one run, by name."""
    return dict(MlflowClient().get_run(run_id).data.metrics)


def run_params(run_id: str) -> dict[str, str]:
    """Every parameter recorded on one run, by name."""
    return dict(MlflowClient().get_run(run_id).data.params)


class RegistrySurvey(NamedTuple):
    """Which registry was actually consulted, and what it holds.

    This exists because of a silence. "No incumbent" is the truth in two very
    different situations - the first model a project ever trains, and a store
    that is simply not the one you meant - and a step that prints only "none
    registered yet" reads identically in both. The second is the dangerous one:
    the gate reports a promotion for a model it never compared to anything.

    Naming the destination and counting what is registered there separates them
    without the reader having to run another command. Nothing here branches on
    which backend the destination happens to be; it reports the URI MLflow
    resolved and the names it found, whatever those turn out to be.
    """

    #: The destination MLflow resolved, exactly as the step used it.
    uri: str
    #: Whether that came from :data:`ENV_TRACKING_URI`, or is MLflow's default.
    from_environment: bool
    #: Every registered model name found there, or ``None`` if listing failed.
    model_names: tuple[str, ...] | None = ()
    #: Why the listing failed, when it did.
    listing_error: str | None = None

    @property
    def is_empty(self) -> bool:
        """True when the store was readable and holds no registered model at all."""
        return self.model_names == ()

    def destination(self) -> str:
        """The URI, and where the choice of it came from."""
        origin = (
            f"{ENV_TRACKING_URI} is set"
            if self.from_environment
            else f"{ENV_TRACKING_URI} is unset, so this is MLflow's default store"
        )
        return f"{self.uri}  ({origin})"

    def contents(self, shown: int = 5) -> str:
        """How many registered models the destination holds, and their names."""
        if self.model_names is None:
            return f"could not be listed ({self.listing_error})"
        if not self.model_names:
            return "0 registered models - this store has never held one"
        listed = ", ".join(self.model_names[:shown])
        if len(self.model_names) > shown:
            listed += f", and {len(self.model_names) - shown} more"
        return f"{len(self.model_names)} registered model(s): {listed}"

    def caution(self, model_name: str) -> str | None:
        """What a reader should check, given what was found, or ``None``.

        The wording is aimed at the person who opened a second terminal: the
        tracking destination is read per process and does not follow them.
        """
        if self.model_names is None:
            return (
                "the destination above would not list its registered models, so "
                '"no incumbent" here is unverified rather than established'
            )
        if not self.model_names:
            unset = not self.from_environment
            return (
                "nothing at all is registered at the destination above. That is what "
                "a project's genuine first run looks like. If you expected to find "
                "something there, the destination is not the one you think: "
                + (
                    f"{ENV_TRACKING_URI} is unset here, and it is read per process, "
                    "so a second terminal does not inherit the export you made in "
                    "the first."
                    if unset
                    else f"{ENV_TRACKING_URI} names it, and a relative path in it "
                    "resolves against the current working directory."
                )
            )
        if model_name in self.model_names:
            return (
                f"the destination above lists {model_name!r} but holds no version of it, "
                "so there was nothing to compare against"
            )
        return (
            f"the destination above holds {len(self.model_names)} registered model(s), "
            f"but none named {model_name!r} - check --model-name against the names above"
        )


def survey_registry(page_size: int = 200, max_pages: int = 50) -> RegistrySurvey:
    """Read back the destination consulted and the registered models it holds.

    A diagnostic, so it declines to fail: a store that will not list its models
    is reported as such rather than raising over the answer the caller actually
    came for.
    """
    survey = RegistrySurvey(uri=mlflow.get_tracking_uri(), from_environment=_tracking_uri_is_set())
    try:
        return survey._replace(model_names=_registered_model_names(page_size, max_pages))
    except Exception as error:  # pragma: no cover - depends on the backend
        return survey._replace(model_names=None, listing_error=f"{type(error).__name__}: {error}")


def _tracking_uri_is_set() -> bool:
    """Whether the operator named a destination, rather than taking the default."""
    return bool(os.environ.get(ENV_TRACKING_URI, "").strip())


def _registered_model_names(page_size: int, max_pages: int) -> tuple[str, ...]:
    """Every registered model name at the current destination, paging through."""
    client = MlflowClient()
    names: list[str] = []
    token = None
    for _ in range(max_pages):
        page = client.search_registered_models(max_results=page_size, page_token=token)
        names.extend(model.name for model in page)
        token = getattr(page, "token", None)
        if not token:
            break
    return tuple(names)


def find_incumbent(model_name: str) -> Incumbent | None:
    """The latest version of ``model_name``, with the metrics of the run behind it.

    ``None`` means nothing has been registered under that name yet - the first
    time this pipeline is ever run, and the case the quality gate has to survive.

    The *latest* version is the incumbent rather than the *best* one, and that is
    a consequence of the gate rather than a shortcut: a version only exists
    because the gate promoted it, so the newest version is the best one that has
    been seen so far under the policy in force at the time.
    """
    versions = MlflowClient().search_model_versions(f"name = '{model_name}'")
    if not versions:
        return None
    latest = max(versions, key=lambda version: int(version.version))
    if not latest.run_id:
        raise RuntimeError(
            f"version {latest.version} of {model_name!r} has no run behind it, so its "
            "metrics cannot be read; the gate has nothing to compare against"
        )
    return Incumbent(
        version=int(latest.version),
        run_id=latest.run_id,
        metrics=run_metrics(latest.run_id),
    )


def record_gate_decision(run_id: str, *, promote: bool, reason: str, metric: str) -> str:
    """Tag ``run_id`` with the gate's verdict, and return the verdict recorded."""
    verdict = PROMOTED if promote else REJECTED
    client = MlflowClient()
    client.set_tag(run_id, GATE_DECISION_TAG, verdict)
    client.set_tag(run_id, GATE_REASON_TAG, reason)
    client.set_tag(run_id, GATE_METRIC_TAG, metric)
    return verdict


def read_gate_decision(run_id: str) -> str | None:
    """The verdict the evaluate step left on ``run_id``, or ``None`` if it never ran."""
    return MlflowClient().get_run(run_id).data.tags.get(GATE_DECISION_TAG)
