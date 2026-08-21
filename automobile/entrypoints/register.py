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
"""

import argparse
import sys

import mlflow

from automobile.entrypoints.train import MODEL_ARTIFACT_NAME
from automobile.tracking import (
    DEFAULT_REGISTERED_MODEL_NAME,
    GATE_DECISION_TAG,
    PROMOTED,
    read_gate_decision,
    run_metrics,
    run_params,
)


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

    verdict = read_gate_decision(args.run_id)
    if verdict != PROMOTED:
        missing = "the evaluate step has not run against it" if verdict is None else verdict
        print(
            f"refusing to register run {args.run_id}: {GATE_DECISION_TAG} is {missing}. "
            "Only a candidate the quality gate promoted is registered.",
            file=sys.stderr,
        )
        return 1

    model_uri = args.model_uri or f"runs:/{args.run_id}/{MODEL_ARTIFACT_NAME}"
    version = mlflow.register_model(model_uri, args.model_name, tags=lineage_tags(args.run_id))

    print(f"registered model: {version.name}")
    print(f"new version:      {version.version}")
    print(f"from run:         {args.run_id}")
    print(f"model uri:        {model_uri}")
    print(f"metrics:          {run_metrics(args.run_id)}")
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised as a subprocess
    sys.exit(main())
