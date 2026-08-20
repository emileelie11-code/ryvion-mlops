"""Tests for the committed seed dataset.

The dataset is committed to the repository rather than downloaded, because a
classroom that depends on an upstream host being reachable at 09:00 is a
classroom that occasionally does not happen. These tests hold the fixture to
what the rest of the course is written against.

The most important assertion here is the one about the sentinel values. Six rows
of the canonical UCI "Auto MPG" data carry ``?`` in the horsepower column
instead of a number. They are not dirt to be scrubbed before committing: they
are the defect the data contract is written against and the defect the model
pipeline has to survive. A pre-cleaned copy of this file would silently remove
the only genuine data-quality problem the course has.
"""

import pandas as pd

from automobile import dataset


def test_the_seed_file_is_committed():
    assert dataset.SEED_DATA_PATH.is_file(), (
        f"the seed dataset must be committed at {dataset.SEED_DATA_PATH}"
    )


def test_the_dataset_has_the_documented_shape():
    frame = dataset.load_dataset()

    assert frame.shape == (398, 9)
    assert tuple(frame.columns) == dataset.COLUMNS


def test_the_target_and_features_together_account_for_every_column():
    assert dataset.TARGET not in dataset.FEATURES
    assert (dataset.TARGET, *dataset.FEATURES) == dataset.COLUMNS
    assert dataset.FREE_TEXT_FEATURE in dataset.FEATURES
    assert dataset.FREE_TEXT_FEATURE not in dataset.NUMERIC_FEATURES


def test_exactly_six_sentinel_values_survive_in_horsepower():
    """The load-bearing defect. Cleaning these away would break the next slice."""
    frame = dataset.load_dataset()

    sentinels = frame == dataset.SENTINEL

    assert sentinels["horsepower"].sum() == 6
    assert sentinels.drop(columns=["horsepower"]).to_numpy().sum() == 0


def test_horsepower_is_not_pre_coerced_to_a_number():
    """A numeric horsepower column would mean the sentinels had been dropped."""
    frame = dataset.load_dataset()

    assert frame["horsepower"].dtype == object
    assert pd.to_numeric(frame["horsepower"], errors="coerce").isna().sum() == 6


def test_the_target_is_positive_everywhere():
    frame = dataset.load_dataset()

    assert (frame[dataset.TARGET] > 0).all()
