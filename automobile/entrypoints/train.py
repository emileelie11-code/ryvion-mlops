"""Step 2 of the training pipeline: fit the model and log the artifact.

Run as ``automobile-train`` or ``python -m automobile.entrypoints.train``.

The estimator is obtained from the model factory rather than constructed here,
so that training, any future retraining and the serving container cannot drift
apart. The artifact logged is a complete pipeline - preprocessing, scaling and
estimator together - carrying its own fitted preprocessing statistics, a
signature and an input example.

The model is logged with ``code_paths``, so the package the fitted pipeline
unpickles into travels *inside* the artifact rather than being an unstated
requirement of whatever environment loads it - and the inferred pip requirement
naming that same package is struck back out, because a model that carries its
code must not also ask an index for it. See
:func:`automobile.model_factory.model_code_paths` for why both halves are
necessary.

The tracking destination is read from the environment and never hard-coded: with
nothing configured the run lands in a local file-based store and needs no cloud
account at all.

This module is an argparse shell. Every decision it makes - how the data is
split, how the model is built, how it is scored - lives in a pure module next
door, which is where the unit tests point.
"""

import argparse
import sys
from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow.models import infer_signature, update_model_requirements

from automobile import dataset
from automobile.metrics import get_model_metrics
from automobile.model_factory import PACKAGE_NAME, build_pipeline, model_code_paths
from automobile.split import DEFAULT_TEST_SIZE, RANDOM_SEED, split_data
from automobile.tracking import configure_tracking

#: The artifact path the model is logged under, inside the run.
MODEL_ARTIFACT_NAME = "model"


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser for the train step."""
    parser = argparse.ArgumentParser(
        prog="automobile-train",
        description="Fit the automobile model pipeline and log it to the tracking store.",
        epilog=(
            "The tracking destination comes from the environment: MLFLOW_TRACKING_URI "
            "selects the backend (unset means a local ./mlruns file store) and "
            "MLFLOW_EXPERIMENT_NAME the experiment. No credentials are needed to run "
            "this locally."
        ),
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=dataset.SEED_DATA_PATH,
        help="CSV to train on (default: the committed seed dataset).",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=DEFAULT_TEST_SIZE,
        help="Share of the data held back for testing (default: %(default)s).",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=RANDOM_SEED,
        help="Seed for the split, so that a run is reproducible (default: %(default)s).",
    )
    parser.add_argument(
        "--experiment-name",
        default=None,
        help="Experiment to log under (default: MLFLOW_EXPERIMENT_NAME, else automobile-mpg).",
    )
    parser.add_argument(
        "--run-name",
        default=None,
        help="Name for this run (default: whatever the tracking backend generates).",
    )
    return parser


def select_input_example(features: pd.DataFrame, rows: int = 2) -> pd.DataFrame:
    """Pick the rows that ship with the model as its worked example.

    One ordinary row, and - when the training data has one - one row whose
    measurement is missing. The example is part of the artifact's documentation,
    and it is what MLflow infers the signature's nullability from, so it must
    contain a hole: that is what makes ``horsepower`` a double a caller may send
    as JSON ``null`` rather than one they must always know a number for.

    The hole is looked for as a missing value rather than as the string ``?``,
    because the ``?`` sentinel is parsed at the data boundary - see
    :func:`automobile.dataset.parse_sentinels` - and never reaches this far.
    """
    missing = features[dataset.SENTINEL_COLUMN].isna()
    incomplete_row = features[missing].head(1)
    complete_rows = features[~missing]
    if incomplete_row.empty:
        return complete_rows.head(rows)
    return pd.concat([complete_rows.head(rows - 1), incomplete_row])


def main(argv: list[str] | None = None) -> int:
    """Parse arguments and run the train step."""
    args = build_parser().parse_args(argv)

    frame = dataset.load_dataset(args.data)
    split = split_data(frame, test_size=args.test_size, random_state=args.random_seed)

    pipeline = build_pipeline()
    pipeline.fit(split.x_train, split.y_train)

    train_metrics = get_model_metrics(pipeline, split.x_train, split.y_train)
    test_metrics = get_model_metrics(pipeline, split.x_test, split.y_test)

    destination = configure_tracking(args.experiment_name)
    input_example = select_input_example(split.x_train)
    signature = infer_signature(input_example, pipeline.predict(input_example))

    with mlflow.start_run(run_name=args.run_name) as run:
        mlflow.log_params(
            {
                "data_source": str(args.data),
                "rows": len(frame),
                "rows_train": len(split.x_train),
                "rows_test": len(split.x_test),
                "test_size": args.test_size,
                "random_seed": args.random_seed,
                "estimator": type(pipeline.named_steps["model"]).__name__,
                "pipeline_steps": ", ".join(name for name, _ in pipeline.steps),
                "features": ", ".join(dataset.FEATURES),
            }
        )
        mlflow.log_metrics({f"train_{name}": value for name, value in train_metrics.items()})
        mlflow.log_metrics({f"test_{name}": value for name, value in test_metrics.items()})
        model_info = mlflow.sklearn.log_model(
            pipeline,
            name=MODEL_ARTIFACT_NAME,
            signature=signature,
            input_example=input_example,
            code_paths=model_code_paths(),
        )
        # The package is inside the artifact now, so the artifact must stop
        # claiming it as a pip dependency: MLflow infers requirements from what
        # loading the model imports, and this package is not on any index. Left
        # in, it would fail the environment build of any deployment that
        # installs what the model asks for - which is exactly the class of
        # failure code_paths is here to remove.
        update_model_requirements(model_info.model_uri, "remove", [PACKAGE_NAME])
        run_id = run.info.run_id

    print(f"tracking uri:    {destination.uri}")
    print(f"experiment:      {destination.experiment_name}")
    print(f"run id:          {run_id}")
    print(f"model uri:       {model_info.model_uri}")
    print(f"train metrics:   {train_metrics}")
    print(f"test metrics:    {test_metrics}")
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised as a subprocess
    sys.exit(main())
