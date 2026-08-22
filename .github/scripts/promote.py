"""Promotion: move the ``champion`` alias onto the version this run registered.

Run by the fifth job of ``.github/workflows/pipeline.yml``, on the far side of
the ``production`` environment's approval gate. Nothing calls it until a human
has clicked **Approve**, and that is the entire point of the step.

Why this is not a fifth entrypoint
----------------------------------
``validate``, ``train``, ``evaluate`` and ``register`` are the pipeline. They
are commands a person types, they live in ``automobile/``, and they are unit
tested. This is not one of them. It is the *deployment* half of the story, and
it exists so that the approval gate has something real to guard - a job that
prints "approved!" and exits 0 is theatre, and a student is right to ignore
theatre. So the promotion does the smallest real thing available in a repository
with no cloud in it: it names a version.

What an alias is, and why naming a version is a deployment
----------------------------------------------------------
Registration and promotion are different decisions, and the registry gives them
different mechanisms. ``register`` adds a *version*: an automatic consequence of
a candidate clearing the quality gate, and "version 7 exists" says nothing about
whether anyone should serve it. An *alias* is a name that points at exactly one
version, and moving it is a choice somebody makes.

``serving/loader.py`` loads any URI MLflow can resolve, including
``models:/automobile-mpg@champion``. Point the container at that, and this
script - not the training run, not the registry's version counter - decides what
gets served. Which is why the approval sits in front of it and not in front of
``register``.

The alias is also why this step prints what it *replaced*. A promotion that
cannot say which version just stopped being the champion is a deployment with no
rollback story, and here the previous version number is the whole of the
rollback story: one call points the alias back at it.

Addressing the version
----------------------
By the run that produced it, never by "the highest version number". In a CI run
the registry starts empty and the highest version is trivially the right one -
which is exactly why leaning on it is a habit worth not forming. The same code
against a shared registry would promote whatever another pipeline happened to
register a second earlier. The training run id travels from job 2 to job 5 as a
job output for this reason, and the version carries it as the
``automobile.run_id`` lineage tag that ``register`` wrote onto it.
"""

import argparse
import sys

from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient

from automobile.tracking import DEFAULT_REGISTERED_MODEL_NAME, survey_registry

#: The alias the serving stack resolves. One name, one meaning: the version a
#: human said should be served.
DEFAULT_ALIAS = "champion"


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser for the promotion step."""
    parser = argparse.ArgumentParser(
        prog="promote",
        description="Point a registry alias at the version a given training run produced.",
        epilog=(
            "Refuses, with a non-zero exit, when the store holds no version for that run - "
            "which is what a promotion aimed at the wrong tracking store looks like."
        ),
    )
    parser.add_argument(
        "--run-id",
        required=True,
        help="The training run whose registered version should become the champion.",
    )
    parser.add_argument(
        "--model-name",
        default=DEFAULT_REGISTERED_MODEL_NAME,
        help="The registered model to move the alias on (default: %(default)s).",
    )
    parser.add_argument(
        "--alias",
        default=DEFAULT_ALIAS,
        help="The alias to move (default: %(default)s).",
    )
    return parser


def version_for_run(client: MlflowClient, model_name: str, run_id: str):
    """The registered version produced by ``run_id``, or ``None``.

    Filtered in Python rather than in the search expression: the filter grammar
    a registry accepts differs between backends, and the number of versions in
    this story is far too small for that difference to be worth inheriting.
    """
    versions = [
        version
        for version in client.search_model_versions(f"name='{model_name}'")
        if version.run_id == run_id
    ]
    if not versions:
        return None
    return max(versions, key=lambda version: int(version.version))


def current_alias_target(client: MlflowClient, model_name: str, alias: str):
    """The version the alias points at now, or ``None`` when it points at nothing.

    An alias that has never been set is the ordinary first-promotion case, not a
    failure, and the registry reports it by raising.
    """
    try:
        return client.get_model_version_by_alias(model_name, alias)
    except MlflowException:
        return None


def main(argv: list[str] | None = None) -> int:
    """Parse arguments and move the alias."""
    args = build_parser().parse_args(argv)
    client = MlflowClient()
    survey = survey_registry()

    candidate = version_for_run(client, args.model_name, args.run_id)
    if candidate is None:
        print(
            f"refusing to promote: {args.model_name!r} has no version registered from run "
            f"{args.run_id} in this store.\n"
            f"tracking store consulted: {survey.destination()}\n"
            f"store holds:              {survey.contents()}\n"
            "A version registered against a different store is invisible from this one.",
            file=sys.stderr,
        )
        return 1

    previous = current_alias_target(client, args.model_name, args.alias)
    client.set_registered_model_alias(args.model_name, args.alias, candidate.version)

    # str() on both, because a version number is an int on some registry
    # backends and a string on others, and this step is not the place to have an
    # opinion about which.
    displaced = f"version {previous.version}" if previous is not None else "(nothing)"

    print(f"promoted model:   {args.model_name}")
    print(f"alias:            {args.alias}")
    print(f"now points at:    version {candidate.version}")
    print(f"previously:       {displaced}")
    print(f"from run:         {args.run_id}")
    print(f"tracking store:   {survey.destination()}")
    print()
    print(f"serve it with:    MODEL_URI=models:/{args.model_name}@{args.alias}")
    if previous is not None:
        print(
            "roll back with:   MlflowClient().set_registered_model_alias("
            f"{args.model_name!r}, {args.alias!r}, {str(previous.version)!r})"
        )
    return 0


if __name__ == "__main__":  # pragma: no cover - only ever run as a workflow step
    sys.exit(main())
