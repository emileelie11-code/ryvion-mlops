"""The round trip this repository's model has to survive: loading without its source.

The fitted pipeline's first stage is a ``FunctionTransformer`` wrapping
:func:`automobile.model_factory.coerce_to_numeric`, and cloudpickle serialises a
module-level function **by reference**. The artifact records the name
``automobile.model_factory.coerce_to_numeric``; unpickling it imports this
package; and anywhere the source tree is absent that import fails. The serving
container met this first and worked around it by copying the package into the
image. A no-code deployment cannot: it builds its own container from the model
registry and has never seen this repository.

So the round trip is demonstrated here rather than asserted. One real training
run is logged into a throwaway tracking store, and the artifact it produces is
loaded **in a subprocess proven unable to import this package by any other
means** - the working directory is elsewhere, the repository is not on
``sys.path``, and every import hook that could still resolve ``automobile`` (the
editable install a developer's virtual environment carries, for one) is torn out
before MLflow is imported. The subprocess proves the negative with ``find_spec``
before it proves the positive with a prediction, and then reports which file the
package it finally imported came from.

A test that merely ran from a different directory while the package stayed
pip-installed would pass against the very defect it exists to catch.
"""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from automobile import dataset
from automobile.entrypoints import train
from automobile.model_factory import PACKAGE_NAME, build_pipeline
from automobile.split import DEFAULT_TEST_SIZE, RANDOM_SEED, split_data

# Run in the subprocess that must not be able to import the package.
#
# It dismantles every route to `automobile` other than the model artifact, and
# refuses to continue if one survives, before MLflow is even imported.
# `PathFinder` is deliberately left in place: it is what resolves the copy
# MLflow prepends to sys.path from inside the artifact, and removing it would
# make this pass for the wrong reason by making the load impossible either way.
LOAD_IN_A_CLEAN_PROCESS = """
import importlib.util
import json
import sys

sys.path = [entry for entry in sys.path if entry not in ("", ".")]


def resolves_the_package(finder):
    try:
        return finder.find_spec({package!r}, None) is not None
    except Exception:
        return False


sys.meta_path[:] = [f for f in sys.meta_path if not resolves_the_package(f)]
importlib.invalidate_caches()

found = importlib.util.find_spec({package!r})
if found is not None:
    raise SystemExit(
        "the package is still importable, so this process proves nothing: " + str(found)
    )

import pandas as pd
import mlflow.pyfunc

model = mlflow.pyfunc.load_model({model_dir!r})
predictions = model.predict(pd.DataFrame(json.loads({records!r})))

import automobile

print(
    "RESULT "
    + json.dumps(
        {{
            "predictions": [float(value) for value in predictions],
            "package_file": automobile.__file__,
        }}
    )
)
"""


def training_data():
    """The exact split the train step fits on, so a local prediction is comparable."""
    split = split_data(
        dataset.load_dataset(), test_size=DEFAULT_TEST_SIZE, random_state=RANDOM_SEED
    )
    return split.x_train, split.y_train


@pytest.fixture(scope="module")
def records() -> list[dict]:
    """A handful of raw rows, the sentinel row included, as a caller would send them."""
    features = dataset.load_dataset()[list(dataset.FEATURES)]
    incomplete = features[features[dataset.SENTINEL_COLUMN].isna()].head(1)
    return json.loads(features.head(4).to_json(orient="records")) + json.loads(
        incomplete.to_json(orient="records")
    )


@pytest.fixture(scope="module")
def logged_model(tmp_path_factory) -> Path:
    """Run the real train step into a throwaway store and return its artifact directory.

    The entrypoint is invoked rather than :func:`mlflow.sklearn.log_model`
    directly, because what is under test is how *this repository* logs its model,
    not what MLflow can be asked to do.
    """
    store = tmp_path_factory.mktemp("tracking-store")
    names = ("MLFLOW_TRACKING_URI", "MLFLOW_EXPERIMENT_NAME")
    previous = {name: os.environ.get(name) for name in names}
    os.environ["MLFLOW_TRACKING_URI"] = store.as_uri()
    os.environ["MLFLOW_EXPERIMENT_NAME"] = "code-paths-round-trip"
    try:
        assert train.main([]) == 0
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value

    artifacts = [path.parent for path in store.rglob("MLmodel")]
    assert len(artifacts) == 1, f"expected exactly one logged model, found {artifacts}"
    return artifacts[0]


@pytest.fixture(scope="module")
def model_away_from_the_repository(logged_model: Path, tmp_path_factory) -> Path:
    """The artifact, copied somewhere with no relationship to this checkout."""
    elsewhere = tmp_path_factory.mktemp("no-source-tree") / "model"
    shutil.copytree(logged_model, elsewhere)
    return elsewhere


@pytest.fixture(scope="module")
def loaded_without_the_source_tree(
    model_away_from_the_repository: Path, records, tmp_path_factory
) -> dict:
    """Load the artifact in a process that cannot import this package, and predict."""
    script = LOAD_IN_A_CLEAN_PROCESS.format(
        package=PACKAGE_NAME,
        model_dir=str(model_away_from_the_repository),
        records=json.dumps(records),
    )
    environment = {
        key: value for key, value in os.environ.items() if key.upper() != "PYTHONPATH"
    }
    environment["PYTHONNOUSERSITE"] = "1"
    environment["MLFLOW_TRACKING_URI"] = tmp_path_factory.mktemp("serving-store").as_uri()

    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=tmp_path_factory.mktemp("nowhere-near-the-repository"),
        env=environment,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "RESULT " in result.stdout, result.stdout + result.stderr
    return json.loads(result.stdout.rsplit("RESULT ", 1)[1])


def test_the_artifact_carries_the_package_it_unpickles_into(logged_model: Path):
    carried = logged_model / "code" / PACKAGE_NAME

    assert carried.is_dir(), sorted(path.name for path in logged_model.iterdir())
    assert (carried / "model_factory.py").is_file()
    assert (carried / "dataset.py").is_file(), "the factory imports it at module scope"


def test_the_artifact_does_not_also_name_the_package_as_a_dependency(logged_model: Path):
    """Carrying the code and asking an index for it are alternatives, not a pair."""
    declared = (logged_model / "requirements.txt").read_text().splitlines()
    declared += (logged_model / "conda.yaml").read_text().splitlines()

    named = [line for line in declared if PACKAGE_NAME in line]

    assert not named, f"{PACKAGE_NAME} is on no package index, so this cannot be met: {named}"


def test_it_predicts_where_the_package_cannot_be_imported(loaded_without_the_source_tree, records):
    predictions = loaded_without_the_source_tree["predictions"]

    assert len(predictions) == len(records)
    assert all(isinstance(value, float) for value in predictions)


def test_the_package_it_imported_came_from_inside_the_artifact(
    loaded_without_the_source_tree, model_away_from_the_repository: Path
):
    """Not from site-packages, not from the checkout - from the model directory."""
    imported = Path(loaded_without_the_source_tree["package_file"])

    assert imported.is_relative_to(model_away_from_the_repository), imported


def test_predictions_are_the_same_with_and_without_the_source_tree(
    loaded_without_the_source_tree, records
):
    """The artifact is a delivery mechanism, not a second implementation."""
    here = build_pipeline().fit(*training_data()).predict(pd.DataFrame(records))

    assert loaded_without_the_source_tree["predictions"] == pytest.approx(list(here))
