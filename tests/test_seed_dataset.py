"""Tests for the committed seed dataset, and for the boundary it is read across.

The dataset is committed to the repository rather than downloaded, because a
classroom that depends on an upstream host being reachable at 09:00 is a
classroom that occasionally does not happen. These tests hold the fixture to
what the rest of the course is written against.

Two things are being held here, and it is worth being clear which is which.

*The committed file* must keep its defect. Six rows of the canonical UCI "Auto
MPG" data carry ``?`` in the horsepower column instead of a number. They are not
dirt to be scrubbed before committing: they are what the data contract is
written against and what the model pipeline has to survive. A pre-cleaned copy of
the CSV would silently remove the only genuine data-quality problem the course
has, so the assertions about the file read it directly with pandas.

*The loader* must not pass that defect on. ``?`` is a 1983 fixed-width-file
encoding artifact, and :func:`automobile.dataset.parse_sentinels` reads it as
the missing value it stands for. That is what makes the model's declared input
schema a nullable ``double`` rather than a ``string`` - so the tests about
``load_dataset`` assert the opposite of the tests about the file, on purpose.
"""

import numpy as np
import pandas as pd
import pytest
from mlflow.models import infer_signature

from automobile import dataset


@pytest.fixture
def committed_file() -> pd.DataFrame:
    """The seed CSV exactly as it sits in the repository, read by plain pandas."""
    return pd.read_csv(dataset.SEED_DATA_PATH)


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


def test_exactly_six_sentinel_values_survive_in_the_committed_file(committed_file):
    """The load-bearing defect. Cleaning it out of the CSV would break the contract."""
    sentinels = committed_file == dataset.SENTINEL

    assert sentinels["horsepower"].sum() == 6
    assert sentinels.drop(columns=["horsepower"]).to_numpy().sum() == 0


def test_horsepower_is_not_pre_coerced_in_the_committed_file(committed_file):
    """A numeric horsepower column in the CSV would mean the sentinels had been dropped."""
    assert committed_file["horsepower"].dtype == object
    assert pd.to_numeric(committed_file["horsepower"], errors="coerce").isna().sum() == 6


def test_the_target_is_positive_everywhere():
    frame = dataset.load_dataset()

    assert (frame[dataset.TARGET] > 0).all()


def test_the_loader_reads_the_sentinel_as_a_missing_value():
    """The boundary parse: ``?`` in the file, ``NaN`` in the frame training sees."""
    frame = dataset.load_dataset()

    assert not (frame == dataset.SENTINEL).to_numpy().any()
    assert frame[dataset.SENTINEL_COLUMN].isna().sum() == 6


def test_the_loader_leaves_horsepower_a_number_rather_than_a_string():
    frame = dataset.load_dataset()

    assert frame[dataset.SENTINEL_COLUMN].dtype == np.float64


def test_the_loader_changes_nothing_but_the_sentinels(committed_file):
    """No imputation, no dropped rows, no renamed columns - one parse and no more."""
    frame = dataset.load_dataset()

    assert frame.shape == committed_file.shape
    assert tuple(frame.columns) == tuple(committed_file.columns)
    unaffected = [column for column in dataset.COLUMNS if column != dataset.SENTINEL_COLUMN]
    pd.testing.assert_frame_equal(frame[unaffected], committed_file[unaffected])


def test_parsing_an_already_parsed_frame_does_nothing_further():
    once = dataset.load_dataset()

    twice = dataset.parse_sentinels(once)

    pd.testing.assert_frame_equal(once, twice)


def test_a_value_that_is_neither_a_number_nor_the_sentinel_is_left_alone():
    """Unknown junk stays put, so that the contract can report it against its row."""
    raw = pd.DataFrame({"horsepower": ["130.0", "?", "n/a"]})

    parsed = dataset.parse_sentinels(raw)

    assert parsed["horsepower"].dtype == object, "an unparseable column stays as it arrived"
    assert parsed["horsepower"].isna().sum() == 1, "only the known sentinel becomes missing"
    assert "n/a" in parsed["horsepower"].tolist()


def test_parsing_does_not_modify_the_frame_it_was_given():
    raw = pd.read_csv(dataset.SEED_DATA_PATH)
    before = raw.copy(deep=True)

    dataset.parse_sentinels(raw)

    pd.testing.assert_frame_equal(raw, before)


def test_the_frame_the_loader_returns_declares_horsepower_as_a_nullable_double():
    """The reason the boundary parse exists: the model's schema, as callers see it.

    A ``string`` here is what forced a scoring client to send ``"130.0"`` and gave
    it no way at all to say "unknown". A nullable ``double`` lets it send a number,
    or JSON ``null``, which the model's own fitted imputer then fills.
    """
    frame = dataset.load_dataset()
    example = frame[list(dataset.FEATURES)].head(1)
    with_a_hole = frame[frame[dataset.SENTINEL_COLUMN].isna()][list(dataset.FEATURES)].head(1)

    signature = infer_signature(pd.concat([example, with_a_hole]))
    horsepower = signature.inputs.input_dict()[dataset.SENTINEL_COLUMN]

    assert horsepower.type.name == "double", "a string here is the defect this slice removed"
    assert not horsepower.required, "a column with a hole in it must be nullable"
