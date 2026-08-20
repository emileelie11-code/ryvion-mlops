"""Behaviour tests for the deterministic split.

The predecessor called ``train_test_split`` with no seed, so two runs of the same
code on the same data produced two different models and two different metrics,
and there was no way to tell a real regression from resampling noise. The seed
is therefore part of the interface, and these tests assert the properties a
caller depends on: the partitions are disjoint, they are exhaustive, the ratio is
honoured, and the same seed gives the same partition.
"""

import pandas as pd
import pytest

from automobile import dataset
from automobile.split import DEFAULT_TEST_SIZE, RANDOM_SEED, split_data


@pytest.fixture
def frame() -> pd.DataFrame:
    """A hundred rows, distinguishable from one another by their index."""
    return pd.DataFrame(
        {
            dataset.TARGET: [float(i) + 1.0 for i in range(100)],
            "cylinders": [4] * 100,
            "displacement": [120.0] * 100,
            "horsepower": ["90.00"] * 100,
            "weight": [2500] * 100,
            "acceleration": [15.0] * 100,
            "model year": [76] * 100,
            "origin": [1] * 100,
            "car name": [f"car {i}" for i in range(100)],
        }
    )


def test_the_features_exclude_the_target_and_keep_every_other_column(frame):
    split = split_data(frame)

    assert tuple(split.x_train.columns) == dataset.FEATURES
    assert tuple(split.x_test.columns) == dataset.FEATURES
    assert split.y_train.name == dataset.TARGET


def test_the_partitions_are_disjoint(frame):
    split = split_data(frame)

    assert set(split.x_train.index).isdisjoint(split.x_test.index)


def test_the_partitions_are_exhaustive(frame):
    split = split_data(frame)

    assert set(split.x_train.index) | set(split.x_test.index) == set(frame.index)
    assert len(split.x_train) + len(split.x_test) == len(frame)


def test_features_and_target_stay_aligned_row_by_row(frame):
    split = split_data(frame)

    for features, target in ((split.x_train, split.y_train), (split.x_test, split.y_test)):
        assert list(features.index) == list(target.index)
        expected = [frame.loc[index, dataset.TARGET] for index in features.index]
        assert list(target) == expected


def test_the_requested_ratio_is_honoured(frame):
    split = split_data(frame, test_size=0.25)

    assert len(split.x_test) == 25
    assert len(split.x_train) == 75


def test_the_default_ratio_is_honoured(frame):
    split = split_data(frame)

    assert len(split.x_test) == int(len(frame) * DEFAULT_TEST_SIZE)


def test_the_same_seed_produces_the_same_partition(frame):
    first = split_data(frame, random_state=RANDOM_SEED)
    second = split_data(frame, random_state=RANDOM_SEED)

    assert list(first.x_train.index) == list(second.x_train.index)
    assert list(first.x_test.index) == list(second.x_test.index)


def test_a_different_seed_produces_a_different_partition(frame):
    default = split_data(frame, random_state=RANDOM_SEED)
    other = split_data(frame, random_state=RANDOM_SEED + 1)

    assert list(default.x_test.index) != list(other.x_test.index)


def test_the_split_does_not_mutate_the_frame_it_was_given(frame):
    before = frame.copy(deep=True)

    split_data(frame)

    pd.testing.assert_frame_equal(frame, before)
