"""Behaviour tests for the model factory.

The factory is the load-bearing extraction of this repository: training, any
future retraining and the serving container all obtain the estimator from it, so
the preprocessing cannot diverge between them. That is what closes the
training/serving skew the predecessor carried, where imputation ran outside the
model and the caller of the scoring service was expected to reproduce statistics
it had never seen.

These tests therefore assert on what a caller of the pipeline can observe - that
it fits on the raw data with its sentinel values intact, that it predicts from
raw named columns, and that its fitted statistics travel with it through a
serialisation round trip. None of them assert a memorised number.
"""

import pickle

import numpy as np
import pandas as pd
import pytest
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from automobile import dataset
from automobile.model_factory import build_pipeline

RAW_ROWS = [
    (18.0, 8, 307.0, "130.0", 3504, 12.0, 70, 1, "chevrolet chevelle malibu"),
    (15.0, 8, 350.0, "165.0", 3693, 11.5, 70, 1, "buick skylark 320"),
    (24.0, 4, 113.0, "95.00", 2372, 15.0, 70, 3, "toyota corona mark ii"),
    (25.0, 4, 98.00, "?", 2046, 19.0, 71, 1, "ford pinto"),
    (32.0, 4, 91.00, "67.00", 1965, 15.7, 82, 3, "honda civic"),
    (44.3, 4, 90.00, "48.00", 2085, 21.7, 80, 2, "vw rabbit c (diesel)"),
]


@pytest.fixture
def raw_frame() -> pd.DataFrame:
    """A frame shaped exactly like the seed data, sentinel value included."""
    return pd.DataFrame(RAW_ROWS, columns=list(dataset.COLUMNS))


@pytest.fixture
def features(raw_frame: pd.DataFrame) -> pd.DataFrame:
    return raw_frame[list(dataset.FEATURES)]


@pytest.fixture
def target(raw_frame: pd.DataFrame) -> pd.Series:
    return raw_frame[dataset.TARGET]


def test_the_factory_returns_a_pipeline_with_preprocessing_scaling_and_an_estimator():
    pipeline = build_pipeline()

    assert isinstance(pipeline, Pipeline)
    assert [name for name, _ in pipeline.steps] == ["prep", "scale", "model"]
    assert isinstance(pipeline.named_steps["scale"], StandardScaler)
    assert isinstance(pipeline.named_steps["model"], LinearRegression)


def test_the_pipeline_comes_back_unfitted(features):
    pipeline = build_pipeline()

    with pytest.raises(NotFittedError):
        pipeline.predict(features)


def test_each_call_returns_an_independent_pipeline(features, target):
    first = build_pipeline()
    second = build_pipeline()

    assert first is not second
    first.fit(features, target)

    with pytest.raises(NotFittedError):
        second.predict(features)


def test_it_fits_on_raw_data_without_any_pre_cleaning(features, target):
    """No caller-side coercion, imputation or column dropping happens first."""
    pipeline = build_pipeline().fit(features, target)

    predictions = pipeline.predict(features)

    assert predictions.shape == (len(features),)
    assert np.isfinite(predictions).all()


def test_it_predicts_from_a_single_raw_row_containing_the_sentinel(features, target):
    pipeline = build_pipeline().fit(features, target)
    sentinel_row = features[features["horsepower"] == dataset.SENTINEL]

    assert not sentinel_row.empty, "the fixture must contain a sentinel row"

    prediction = pipeline.predict(sentinel_row)

    assert prediction.shape == (1,)
    assert np.isfinite(prediction).all()


def test_the_sentinel_is_imputed_with_a_statistic_learned_at_fit_time(features, target):
    """The caller does not supply the imputation value - the artifact carries it."""
    pipeline = build_pipeline().fit(features, target)
    training_mean = pd.to_numeric(features["horsepower"], errors="coerce").mean()

    sentinel_row = features[features["horsepower"] == dataset.SENTINEL]
    filled_row = sentinel_row.assign(horsepower=str(training_mean))

    assert pipeline.predict(sentinel_row) == pytest.approx(pipeline.predict(filled_row))


def test_the_free_text_column_cannot_influence_a_prediction(features, target):
    pipeline = build_pipeline().fit(features, target)
    renamed = features.assign(**{dataset.FREE_TEXT_FEATURE: "a car that does not exist"})

    assert pipeline.predict(features) == pytest.approx(pipeline.predict(renamed))


def test_a_numeric_horsepower_is_accepted_as_well_as_a_string_one(features, target):
    """Coercion is inside the model, so both representations reach the same result."""
    pipeline = build_pipeline().fit(features, target)
    numeric = features.assign(horsepower=pd.to_numeric(features["horsepower"], errors="coerce"))

    assert pipeline.predict(features) == pytest.approx(pipeline.predict(numeric))


def test_fitted_preprocessing_statistics_survive_a_serialisation_round_trip(features, target):
    pipeline = build_pipeline().fit(features, target)

    restored = pickle.loads(pickle.dumps(pipeline))

    assert restored.predict(features) == pytest.approx(pipeline.predict(features))
    sentinel_row = features[features["horsepower"] == dataset.SENTINEL]
    assert restored.predict(sentinel_row) == pytest.approx(pipeline.predict(sentinel_row))


def test_scaling_statistics_are_fitted_rather_than_assumed(features, target):
    pipeline = build_pipeline().fit(features, target)
    scaler = pipeline.named_steps["scale"]

    assert scaler.mean_.shape == (len(dataset.NUMERIC_FEATURES),)
    assert (scaler.scale_ > 0).all()
