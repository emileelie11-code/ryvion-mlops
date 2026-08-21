"""Step 4 of the training pipeline: register the promoted model.

Run as ``automobile-register`` or ``python -m automobile.entrypoints.register``.

The step runs only when the quality gate promoted the candidate, and it records
what the model was trained on so that a model's lineage identifies the data that
produced it.

"Only when the gate promoted it" is enforced here rather than assumed. In a
pipeline the evaluate step's non-zero exit already stops the run before this job
starts, but that guarantee belongs to the orchestrator, and this step is also a
command a person can type. So it reads the verdict the evaluate step tagged onto
the run and refuses anything that is not a promotion - including a run that was
never evaluated at all. There is no override flag: a gate with a bypass is not a
gate, and the no-incumbent case means the first model ever trained is promoted
anyway.

Registration needs a registry, and a registry needs a database behind it rather
than the ``./mlruns`` directory that is enough for recording runs. Locally that
is ``MLFLOW_TRACKING_URI=sqlite:///mlflow.db`` - configuration, not code, and no
account. The README spells it out.

Which store, though, is a question this step answers out loud rather than
assumes. Registering into a throwaway store looks exactly like registering into
the project's own: a version number comes back either way. So the destination
consulted and what it already held are printed on every path, including the
refusal - a run that "was never evaluated" is just as often a run that was
evaluated somewhere else - and a version 1 says so loudly instead of blending in
with version 4.
"""

import argparse
import sys
import textwrap

import mlflow

from automobile.entrypoints.train import MODEL_ARTIFACT_NAME
from automobile.tracking import (
    DEFAULT_REGISTERED_MODEL_NAME,
    GATE_DECISION_TAG,
    PROMOTED,
    read_gate_decision,
    run_metrics,
    run_params,
    survey_registry,
)

#: The register step's equivalent of the evaluate step's loud first-ever marker.
#: A version 1 in the wrong store reads exactly like a version 1 in the right
#: one, and only the count of what was already there tells them apart.
FIRST_VERSION_HERE = "*** FIRST VERSION IN THIS STORE ***"


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser for the register step."""
    parser = argparse.ArgumentParser(
        prog="automobile-register",
        description="Register the promoted model against the data asset it was trained on.",
        epilog=(
            "Refuses, with a non-zero exit, any run the evaluate step did not tag as "
            "promoted. Needs a database-backed tracking destination for the registry: "
            "locally, MLFLOW_TRACKING_URI=sqlite:///mlflow.db."
        ),
    )
    parser.add_argument(
        "--run-id",
        required=True,
        help="The training run holding the model and the gate's verdict.",
    )
    parser.add_argument(
        "--model-name",
        default=DEFAULT_REGISTERED_MODEL_NAME,
        help="Name to register the model under (default: %(default)s).",
    )
    parser.add_argument(
        "--model-uri",
        default=None,
        help=(
            "Model to register (default: the model logged by that run, "
            f"runs:/RUN_ID/{MODEL_ARTIFACT_NAME})."
        ),
    )
    return parser


def lineage_tags(run_id: str) -> dict[str, str]:
    """What this version was made from, recorded on the version itself.

    A registered version that cannot say which run and which data produced it is
    a model nobody can reproduce. Locally the data is a path to the seed file;
    on the platform the same parameter carries a versioned data asset, and this
    tag is where it lands.
    """
    params = run_params(run_id)
    tags = {"automobile.run_id": run_id}
    if "data_source" in params:
        tags["automobile.data_source"] = params["data_source"]
    if "rows" in params:
        tags["automobile.rows"] = params["rows"]
    return tags


def main(argv: list[str] | None = None) -> int:
    """Parse arguments and run the register step."""
    args = build_parser().parse_args(argv)

    survey = survey_registry()

    verdict = read_gate_decision(args.run_id)
    if verdict != PROMOTED:
        missing = "the evaluate step has not run against it" if verdict is None else verdict
        print(
            f"refusing to register run {args.run_id}: {GATE_DECISION_TAG} is {missing}. "
            "Only a candidate the quality gate promoted is registered.\n"
            f"tracking store consulted: {survey.destination()}\n"
            "A run evaluated against a different store carries no verdict in this one.",
            file=sys.stderr,
        )
        return 1

    model_uri = args.model_uri or f"runs:/{args.run_id}/{MODEL_ARTIFACT_NAME}"
    version = mlflow.register_model(model_uri, args.model_name, tags=lineage_tags(args.run_id))

    print(f"registered model: {version.name}")
    print(f"new version:      {version.version}")
    print(f"from run:         {args.run_id}")
    print(f"model uri:        {model_uri}")
    print(f"tracking store:   {survey.destination()}")
    print(f"store held:       {survey.contents()} (before this registration)")
    print(f"metrics:          {run_metrics(args.run_id)}")

    if str(version.version) == "1":
        print()
        print(f"{FIRST_VERSION_HERE} Nothing was registered under")
        print(f"{args.model_name!r} at that destination until now.")
        caution = survey.caution(args.model_name)
        if caution:
            print(textwrap.fill(caution, width=88, initial_indent="  ", subsequent_indent="  "))
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised as a subprocess
    sys.exit(main())
