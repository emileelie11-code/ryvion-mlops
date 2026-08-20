"""Deterministic partitioning of the dataset into a training and a test half.

The seed is part of the interface rather than a detail. The predecessor called
``train_test_split`` without one, so two runs of unchanged code produced two
different models and two different metrics, and a genuine regression was
indistinguishable from resampling noise. A quality gate built on top of that
would be measuring the random number generator.
"""

from typing import NamedTuple

import pandas as pd
from sklearn.model_selection import train_test_split

from automobile.dataset import FEATURES, TARGET

#: The default seed. Fixed, published and boring, so that a run is reproducible.
RANDOM_SEED = 42

#: The default share of the data held back for testing.
DEFAULT_TEST_SIZE = 0.2


class Split(NamedTuple):
    """The four frames a training run needs, with their row order preserved."""

    x_train: pd.DataFrame
    x_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series


def split_data(
    frame: pd.DataFrame,
    *,
    test_size: float = DEFAULT_TEST_SIZE,
    random_state: int = RANDOM_SEED,
) -> Split:
    """Split ``frame`` into features and target, and each of those into two parts.

    The features come back as named columns in the dataset's own order, free-text
    column included: the model pipeline drops what it does not want, so what is
    handed to it here is a row exactly as the data has it.
    """
    features = frame[list(FEATURES)]
    target = frame[TARGET]
    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=test_size,
        random_state=random_state,
    )
    return Split(x_train=x_train, x_test=x_test, y_train=y_train, y_test=y_test)
