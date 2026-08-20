"""Behaviour tests for metric computation.

The inherited test suite is the anti-pattern this file exists to replace. Its
single test asserted that a metric equalled ``0.029843893480257067`` - a
fingerprint of one particular fit, which says nothing about correctness, breaks
on any harmless change, and (because the CI template that would have run it was
commented out) never once executed.

The properties below are what actually matter: the expected keys are present, a
perfect predictor scores zero error, and error grows as predictions get worse.
They hold for any regression model, so they survive changes to the estimator.
"""

import numpy as np
import pandas as pd
import pytest

from automobile.metrics import METRIC_NAMES, get_model_metrics


class ConstantOffsetModel:
    """A stand-in model that predicts the truth, displaced by a fixed amount."""

    def __init__(self, offset: float) -> None:
        self.offset = offset

    def fit(self, features, target):
        self._target = np.asarray(target, dtype=float)
        return self

    def predict(self, features):
        return self._target + self.offset


@pytest.fixture
def features() -> pd.DataFrame:
    return pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0, 5.0]})


@pytest.fixture
def target() -> pd.Series:
    return pd.Series([10.0, 12.0, 14.0, 16.0, 18.0], name="mpg")


def test_the_expected_metric_keys_are_present(features, target):
    model = ConstantOffsetModel(1.0).fit(features, target)

    metrics = get_model_metrics(model, features, target)

    assert set(metrics) == set(METRIC_NAMES)
    assert all(isinstance(value, float) for value in metrics.values())


def test_a_perfect_predictor_scores_zero_error(features, target):
    model = ConstantOffsetModel(0.0).fit(features, target)

    metrics = get_model_metrics(model, features, target)

    assert metrics["mse"] == pytest.approx(0.0)
    assert metrics["rmse"] == pytest.approx(0.0)
    assert metrics["mae"] == pytest.approx(0.0)
    assert metrics["r2"] == pytest.approx(1.0)


def test_error_is_monotonic_in_prediction_quality(features, target):
    errors = [
        get_model_metrics(ConstantOffsetModel(offset).fit(features, target), features, target)
        for offset in (0.0, 1.0, 2.0, 5.0)
    ]

    for name in ("mse", "rmse", "mae"):
        values = [metrics[name] for metrics in errors]
        assert values == sorted(values)
        assert len(set(values)) == len(values), f"{name} must distinguish these predictors"

    r2 = [metrics["r2"] for metrics in errors]
    assert r2 == sorted(r2, reverse=True)


def test_the_root_mean_squared_error_is_the_root_of_the_mean_squared_error(features, target):
    model = ConstantOffsetModel(3.0).fit(features, target)

    metrics = get_model_metrics(model, features, target)

    assert metrics["rmse"] == pytest.approx(metrics["mse"] ** 0.5)


def test_the_sign_of_the_error_does_not_change_the_score(features, target):
    over = get_model_metrics(ConstantOffsetModel(2.0).fit(features, target), features, target)
    under = get_model_metrics(ConstantOffsetModel(-2.0).fit(features, target), features, target)

    assert over == pytest.approx(under)


def test_metrics_are_a_pure_function_of_their_arguments(features, target):
    model = ConstantOffsetModel(1.5).fit(features, target)
    before = target.copy(deep=True)

    first = get_model_metrics(model, features, target)
    second = get_model_metrics(model, features, target)

    assert first == second
    pd.testing.assert_series_equal(target, before)
