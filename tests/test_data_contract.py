"""Behaviour tests for the data contract.

The contract is the pipeline's step zero, and the thing it has to get right is
not "did it say no" but "did it say *why*, and *where*". These tests therefore
assert on what a caller of :func:`automobile.data_contract.validate` can observe:
that the committed dataset passes, that each rule fails on its own when the
defect it names is introduced, and that the report carries the rule's name and
the offending row numbers.

Nothing here asserts a memorised value out of the report - no error string, no
Pandera internal, no exception type. The corruptions are built by breaking the
real dataset one property at a time, so a rule that stopped working would be
caught by the rule's own test rather than by a fingerprint of one implementation.
"""

import numpy as np
import pandas as pd
import pytest

from automobile import dataset
from automobile.data_contract import (
    EXPECTED_MISSING_HORSEPOWER,
    EXPECTED_ROWS,
    RULE_COLUMNS_PRESENT,
    RULE_COLUMN_TYPES,
    RULE_HORSEPOWER_MISSING_COUNT,
    RULE_HORSEPOWER_NUMERIC,
    RULE_MPG_POSITIVE,
    RULE_NAMES,
    RULE_NO_MISSING_VALUES,
    RULE_NO_UNEXPECTED_COLUMNS,
    RULE_ROW_COUNT,
    RULES,
    validate,
)

#: Rows chosen out of the middle of the file, so that a corruption cannot pass
#: by accident because it happened to land on row zero.
CORRUPTED_ROWS = (3, 41)


@pytest.fixture
def conforming() -> pd.DataFrame:
    """The committed dataset, read the way training reads it."""
    return dataset.load_dataset()


def with_negative_target(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.assign(mpg=frame["mpg"].mask(frame.index.isin(CORRUPTED_ROWS), -1.0))


def with_an_unparsed_sentinel(frame: pd.DataFrame) -> pd.DataFrame:
    """The frame as it looks when somebody reads the CSV without the loader."""
    corrupted = frame.copy()
    corrupted[dataset.SENTINEL_COLUMN] = corrupted[dataset.SENTINEL_COLUMN].astype(object)
    corrupted.loc[list(CORRUPTED_ROWS), dataset.SENTINEL_COLUMN] = dataset.SENTINEL
    return corrupted


def with_a_seventh_hole(frame: pd.DataFrame) -> pd.DataFrame:
    corrupted = frame.copy()
    corrupted.loc[CORRUPTED_ROWS[0], dataset.SENTINEL_COLUMN] = np.nan
    return corrupted


def with_a_missing_car_name(frame: pd.DataFrame) -> pd.DataFrame:
    corrupted = frame.copy()
    corrupted.loc[CORRUPTED_ROWS[0], dataset.FREE_TEXT_FEATURE] = np.nan
    return corrupted


#: One corruption per rule. Every rule the contract publishes appears here, and
#: a test below fails the build if one ever stops appearing.
CORRUPTIONS = {
    RULE_COLUMNS_PRESENT: lambda frame: frame.drop(columns=["weight"]),
    RULE_NO_UNEXPECTED_COLUMNS: lambda frame: frame.assign(scraped_at="2026-08-20"),
    RULE_COLUMN_TYPES: lambda frame: frame.assign(weight=frame["weight"].astype(float)),
    RULE_NO_MISSING_VALUES: with_a_missing_car_name,
    RULE_ROW_COUNT: lambda frame: frame.drop(index=list(CORRUPTED_ROWS)),
    RULE_MPG_POSITIVE: with_negative_target,
    RULE_HORSEPOWER_NUMERIC: with_an_unparsed_sentinel,
    RULE_HORSEPOWER_MISSING_COUNT: with_a_seventh_hole,
}

#: The corruptions that break exactly one rule. The sentinel one is not among
#: them on purpose: a column holding ``?`` is both non-numeric *and* the wrong
#: type, and a contract that hid the second finding would be lying.
ISOLATED = tuple(rule for rule in CORRUPTIONS if rule != RULE_HORSEPOWER_NUMERIC)

#: The rules whose violations point at particular rows rather than at the frame
#: as a whole.
ROW_LEVEL = (RULE_MPG_POSITIVE, RULE_HORSEPOWER_NUMERIC, RULE_NO_MISSING_VALUES)


def test_the_committed_dataset_honours_the_contract(conforming):
    report = validate(conforming)

    assert report.ok, report.summary()
    assert report.violations == ()
    assert report.failed_rules == ()
    assert report.rows_checked == len(conforming)


def test_a_conforming_report_is_truthy_and_says_so(conforming):
    report = validate(conforming)

    assert bool(report) is True
    assert "PASS" in report.summary()


def test_every_published_rule_has_a_description_and_a_test():
    assert set(RULES) == set(RULE_NAMES)
    assert set(CORRUPTIONS) == set(RULE_NAMES), (
        "every rule the contract publishes needs a corruption that provokes it"
    )
    assert all(RULES[rule] for rule in RULE_NAMES), "a rule with no description explains nothing"


@pytest.mark.parametrize("rule", sorted(CORRUPTIONS), ids=lambda rule: rule)
def test_each_rule_fails_when_the_defect_it_names_is_introduced(rule, conforming):
    report = validate(CORRUPTIONS[rule](conforming))

    assert not report.ok
    assert bool(report) is False
    assert rule in report.failed_rules


@pytest.mark.parametrize("rule", sorted(ISOLATED), ids=lambda rule: rule)
def test_each_rule_fails_independently_of_the_others(rule, conforming):
    report = validate(CORRUPTIONS[rule](conforming))

    assert report.failed_rules == (rule,), report.summary()


@pytest.mark.parametrize("rule", sorted(CORRUPTIONS), ids=lambda rule: rule)
def test_the_report_names_the_broken_rule_in_its_summary(rule, conforming):
    report = validate(CORRUPTIONS[rule](conforming))

    summary = report.summary()

    assert "FAIL" in summary
    assert rule in summary
    assert RULES[rule] in summary, "the summary must explain the rule, not just name it"


@pytest.mark.parametrize("rule", sorted(ROW_LEVEL), ids=lambda rule: rule)
def test_a_row_level_rule_identifies_the_rows_it_objected_to(rule, conforming):
    report = validate(CORRUPTIONS[rule](conforming))

    offending = report.rows_for(rule)

    assert offending, f"{rule} must say which rows are wrong"
    assert set(offending) <= set(CORRUPTED_ROWS)
    assert all(str(row) in report.summary() for row in offending)


def test_the_target_rule_finds_every_offending_row_not_merely_the_first(conforming):
    report = validate(with_negative_target(conforming))

    assert report.rows_for(RULE_MPG_POSITIVE) == tuple(sorted(CORRUPTED_ROWS))


def test_a_frame_level_rule_reports_no_rows_rather_than_inventing_one(conforming):
    report = validate(CORRUPTIONS[RULE_ROW_COUNT](conforming))

    assert report.rows_for(RULE_ROW_COUNT) == ()
    assert report.rows_checked == len(conforming) - len(CORRUPTED_ROWS)


def test_a_whole_frame_rule_offers_no_offending_value_rather_than_a_meaningless_one(conforming):
    """"Are there 398 rows?" answers True or False, and neither is a value to print."""
    alone = validate(CORRUPTIONS[RULE_ROW_COUNT](conforming))
    alongside_another_failure = validate(with_negative_target(conforming).drop(index=[100]))

    for report in (alone, alongside_another_failure):
        counted = [v for v in report.violations if v.rule == RULE_ROW_COUNT]
        assert counted, report.summary()
        assert counted[0].values == ()
        assert counted[0].column is None


def test_the_missing_column_is_named_even_though_no_row_is_at_fault(conforming):
    report = validate(CORRUPTIONS[RULE_COLUMNS_PRESENT](conforming))

    assert "weight" in report.summary()


def test_a_cleaned_dataset_fails_just_as_a_corrupted_one_does(conforming):
    """Scrubbing the defect away is a change to the data, and the gate sees it."""
    scrubbed = conforming.assign(horsepower=conforming["horsepower"].fillna(100.0))

    report = validate(scrubbed)

    assert report.failed_rules == (RULE_HORSEPOWER_MISSING_COUNT,)


def test_every_violation_is_reported_in_one_pass_rather_than_one_at_a_time(conforming):
    """A gate that stops at the first defect makes a student run it six times."""
    doubly_broken = with_negative_target(conforming).drop(index=[100, 101])

    report = validate(doubly_broken)

    assert set(report.failed_rules) == {RULE_MPG_POSITIVE, RULE_ROW_COUNT}
    assert report.rows_for(RULE_MPG_POSITIVE) == tuple(sorted(CORRUPTED_ROWS))


def test_validating_a_frame_neither_repairs_nor_disturbs_it(conforming):
    """A contract that quietly fixes what it checks cannot fail, and is not a gate."""
    corrupted = with_an_unparsed_sentinel(conforming)
    before = corrupted.copy(deep=True)

    validate(corrupted)

    pd.testing.assert_frame_equal(corrupted, before)


def test_the_contract_is_a_pure_function_of_the_frame(conforming):
    corrupted = with_negative_target(conforming)

    first = validate(corrupted)
    second = validate(corrupted)

    assert first == second


def test_the_documented_shape_is_the_shape_the_committed_data_has(conforming):
    """The two constants the rules are written around are facts, not guesses."""
    assert len(conforming) == EXPECTED_ROWS
    assert conforming[dataset.SENTINEL_COLUMN].isna().sum() == EXPECTED_MISSING_HORSEPOWER
