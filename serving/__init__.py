"""The hand-built serving application.

This package is the deliberate counterweight to the no-code managed endpoint a
later slice produces. A reader builds this image themselves, reads every line of
it, and runs it on a laptop with no cloud account: two modules, one Dockerfile,
and nothing hidden.

It is intentionally *not* part of the ``automobile`` domain package. Nothing in
here is domain logic - it is an I/O adapter over an HTTP boundary - and keeping
it separate is what lets the serving image carry no training, testing or cloud
tooling at all.

``serving.loader``
    Loading the model through its MLflow flavour, and the input contract the
    model declares for itself. It is not called ``serving.model`` because
    ``serving/model/`` is where the exported artifact itself lands, and a
    module and a directory sharing a name is a puzzle nobody needs to solve.
``serving.app``
    The HTTP surface: a prediction endpoint, a health endpoint and a readiness
    endpoint.
"""
