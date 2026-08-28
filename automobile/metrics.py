"""Metric computation, as a pure function of a model and some data.

Kept separate from training so that the evaluation step, the quality gate and
the tests all ask the same question the same way, and so that the answer can be
asserted on as a set of properties rather than as a remembered number. The test
suite this replaces asserted that one metric equalled ``0.029843893480257067``;
the tests here assert that a perfect predictor scores zero error and that error
grows as predictions get worse, which is true of any regression model and
therefore survives changing one.
"""

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

#: The keys every metric report carries: three error measures, then the score.
METRIC_NAMES = ("mse", "rmse", "mae", "r2")


def get_model_metrics(
    model: Any,
    features: pd.DataFrame,
    target: pd.Series,
) -> dict[str, float]:
    """Score ``model`` on ``features`` and ``target``.

    ``model`` is anything with a ``predict`` method - the fitted pipeline in a
    real run, a stand-in in the tests. Nothing here is written to disk, logged or
    tracked: the caller decides what to do with the numbers.
    """
    predictions = model.predict(features)
    mse = float(mean_squared_error(target, predictions))
    return {
        "mse": mse,
        "rmse": float(np.sqrt(mse)),
        "mae": float(mean_absolute_error(target, predictions)),
        "r2": float(r2_score(target, predictions)),
    }
