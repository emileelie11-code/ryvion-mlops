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

This is a thin adapter over an I/O boundary, so it is not unit tested; the
modules it serves are.
"""

import os
from typing import NamedTuple

import mlflow

#: Used when neither the caller nor the environment names an experiment.
DEFAULT_EXPERIMENT_NAME = "automobile-mpg"

ENV_TRACKING_URI = "MLFLOW_TRACKING_URI"
ENV_EXPERIMENT_NAME = "MLFLOW_EXPERIMENT_NAME"


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
