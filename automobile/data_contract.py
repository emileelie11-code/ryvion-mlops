"""The data contract: every rule about this dataset, behind one call.

``validate(frame)`` returns a :class:`ValidationReport`. That is the whole
interface. Nothing else in this repository is allowed to know that the target
must be positive, that the file has 398 rows, or that ``horsepower`` is the
column with a hole in it - so changing the dataset means changing this module
and :mod:`automobile.dataset`, not hunting through the training script the way
the predecessor's scattered column handling forced.

The rules are written against defects that are genuinely in this data, not
invented ones. Every one of them is a fact about the canonical UCI "Auto MPG"
file as committed:

``expected_columns_are_present`` / ``no_unexpected_columns``
    nine columns, named as the header names them.
``columns_have_the_expected_type``
    the numeric columns are numbers. ``horsepower`` is a *nullable* double, and
    that is the added scope of this slice: the ``?`` sentinel is parsed into a
    missing value at the data boundary (:func:`automobile.dataset.parse_sentinels`)
    so that the model's declared input schema is ``double`` rather than
    ``string``, and JSON ``null`` is how a caller says "unknown".
``no_missing_values_outside_horsepower``
    exactly one column is allowed to have holes in it.
``row_count_is_the_documented_398``
    the seed fixture is a fixture. A run that trains on a different number of
    rows than the course documents is a run whose provenance is in question.
``mpg_is_positive``
    a car does not do a negative number of miles to the gallon.
``horsepower_is_numeric_or_missing``
    the known sentinel is handled at the boundary; anything else that is not a
    number reaches this rule, which names the rows it is in.
``horsepower_missing_count_is_the_documented_six``
    six rows, no more and no fewer. This is the rule that catches sentinel drift
    *through* the loader: a seventh ``?`` is parsed into a seventh missing value
    and fails here.

Pandera rather than a hand-rolled loop, and rather than Great Expectations: the
schema is declarative enough to read on a slide, it reports every violation in
one pass instead of stopping at the first, and it names the offending rows -
which is what makes a failing gate a diagnosis rather than an accusation.

This module is pure. It performs no I/O, imports no cloud SDK and knows nothing
about MLflow; the entrypoint in :mod:`automobile.entrypoints.validate` is the
argparse shell that reads a file, calls this, and turns the report into an exit
code.
"""

from dataclasses import dataclass, field

import pandas as pd
import pandera.pandas as pa
from pandera.errors import SchemaErrors

from automobile.dataset import (
    COLUMNS,
    FREE_TEXT_FEATURE,
    SENTINEL,
    SENTINEL_COLUMN,
    TARGET,
)

#: The name the contract reports itself under.
CONTRACT_NAME = "automobile-mpg"

#: The seed fixture's row count. Documented in the README, asserted in the
#: dataset tests, and enforced here.
EXPECTED_ROWS = 398

#: How many rows carry the ``?`` sentinel - and therefore, after the boundary
#: parse, how many rows have a missing ``horsepower``.
EXPECTED_MISSING_HORSEPOWER = 6

RULE_COLUMNS_PRESENT = "expected_columns_are_present"
RULE_NO_UNEXPECTED_COLUMNS = "no_unexpected_columns"
RULE_COLUMN_TYPES = "columns_have_the_expected_type"
RULE_NO_MISSING_VALUES = "no_missing_values_outside_horsepower"
RULE_ROW_COUNT = "row_count_is_the_documented_398"
RULE_MPG_POSITIVE = "mpg_is_positive"
RULE_HORSEPOWER_NUMERIC = "horsepower_is_numeric_or_missing"
RULE_HORSEPOWER_MISSING_COUNT = "horsepower_missing_count_is_the_documented_six"

#: Every rule, and what a reader of a failing gate needs to be told about it.
RULES: dict[str, str] = {
    RULE_COLUMNS_PRESENT: f"the frame must carry all {len(COLUMNS)} documented columns",
    RULE_NO_UNEXPECTED_COLUMNS: "the frame must carry no column the contract does not name",
    RULE_COLUMN_TYPES: (
        f"every column must hold the type the contract documents - which for "
        f"'{SENTINEL_COLUMN}' is a number, the '{SENTINEL}' sentinel having been parsed "
        "into a missing value on the way in"
    ),
    RULE_NO_MISSING_VALUES: (
        f"only '{SENTINEL_COLUMN}' may hold missing values; every other column must be complete"
    ),
    RULE_ROW_COUNT: f"the dataset must hold exactly {EXPECTED_ROWS} rows",
    RULE_MPG_POSITIVE: f"every '{TARGET}' must be greater than zero",
    RULE_HORSEPOWER_NUMERIC: (
        f"every '{SENTINEL_COLUMN}' must be a number or a missing value; the known "
        f"'{SENTINEL}' sentinel is parsed at the boundary, so anything reaching this "
        "rule is a value nobody has accounted for"
    ),
    RULE_HORSEPOWER_MISSING_COUNT: (
        f"exactly {EXPECTED_MISSING_HORSEPOWER} rows may have a missing "
        f"'{SENTINEL_COLUMN}' - the documented defect, neither grown nor cleaned away"
    ),
}

#: How Pandera's own built-in failures map onto the rule names above. Pandera
#: reports a custom check by the string in its ``error`` argument, which is why
#: every check below is given one; its built-ins need translating.
_BUILT_IN_RULES = {
    "column_in_dataframe": RULE_COLUMNS_PRESENT,
    "column_in_schema": RULE_NO_UNEXPECTED_COLUMNS,
    "not_nullable": RULE_NO_MISSING_VALUES,
}


def _is_numeric_or_missing(series: pd.Series) -> pd.Series:
    """True for every value that is a number or is missing - row by row."""
    return series.isna() | pd.to_numeric(series, errors="coerce").notna()


def _has_the_documented_number_of_holes(series: pd.Series) -> bool:
    return int(series.isna().sum()) == EXPECTED_MISSING_HORSEPOWER


def _has_the_documented_row_count(frame: pd.DataFrame) -> bool:
    return len(frame) == EXPECTED_ROWS


def _build_schema() -> pa.DataFrameSchema:
    """The contract, as a Pandera schema.

    Written out rather than generated from the column tuple: a data contract
    that a reader cannot check against the data by eye is not doing its job.
    """
    numeric = {name: pa.Column(float, nullable=False) for name in ("displacement", "acceleration")}
    integral = {
        name: pa.Column(int, nullable=False)
        for name in ("cylinders", "weight", "model year", "origin")
    }
    return pa.DataFrameSchema(
        name=CONTRACT_NAME,
        columns={
            TARGET: pa.Column(
                float,
                nullable=False,
                checks=pa.Check.gt(0, error=RULE_MPG_POSITIVE),
            ),
            SENTINEL_COLUMN: pa.Column(
                float,
                nullable=True,
                checks=[
                    pa.Check(_is_numeric_or_missing, error=RULE_HORSEPOWER_NUMERIC),
                    pa.Check(
                        _has_the_documented_number_of_holes,
                        # Pandera hides missing values from the checks on a
                        # nullable column; this is the one check whose whole
                        # subject is how many of them there are.
                        ignore_na=False,
                        error=RULE_HORSEPOWER_MISSING_COUNT,
                    ),
                ],
            ),
            FREE_TEXT_FEATURE: pa.Column(str, nullable=False),
            **numeric,
            **integral,
        },
        checks=[pa.Check(_has_the_documented_row_count, error=RULE_ROW_COUNT)],
        # Every column named, and no others. Order is not part of the contract:
        # the loader reads the header by name.
        strict=True,
        ordered=False,
        # Never repair on the way through. A contract that quietly fixes what it
        # was asked to check cannot fail, and a gate that cannot fail is not one.
        coerce=False,
    )


#: The contract itself. Built once; :func:`validate` is a pure function of it.
SCHEMA = _build_schema()

#: Every rule the schema can report, in the order this module documents them.
RULE_NAMES: tuple[str, ...] = tuple(RULES)

#: How many offending rows a violation prints before it starts counting.
ROWS_SHOWN = 8


@dataclass(frozen=True)
class RuleViolation:
    """One rule, broken, and where."""

    rule: str
    column: str | None = None
    rows: tuple[int, ...] = ()
    values: tuple[str, ...] = ()

    @property
    def description(self) -> str:
        """What the rule was asking for, in a sentence."""
        return RULES.get(self.rule, self.rule)

    @property
    def row_count(self) -> int:
        return len(self.rows)

    def __str__(self) -> str:
        where = f" [{self.column}]" if self.column else ""
        lines = [f"{self.rule}{where}", f"    {self.description}"]
        if self.rows:
            shown = ", ".join(str(row) for row in self.rows[:ROWS_SHOWN])
            hidden = self.row_count - ROWS_SHOWN
            more = f" (and {hidden} more)" if hidden > 0 else ""
            lines.append(f"    {self.row_count} offending row(s): {shown}{more}")
        if self.values:
            shown = ", ".join(repr(value) for value in self.values[:ROWS_SHOWN])
            lines.append(f"    offending value(s): {shown}")
        return "\n".join(lines)


@dataclass(frozen=True)
class ValidationReport:
    """The answer to "does this frame honour the contract, and if not, where"."""

    rows_checked: int
    violations: tuple[RuleViolation, ...] = field(default=())

    @property
    def ok(self) -> bool:
        """True when no rule was broken. The gate turns on this."""
        return not self.violations

    def __bool__(self) -> bool:
        return self.ok

    @property
    def failed_rules(self) -> tuple[str, ...]:
        """The name of every rule that was broken, without duplicates."""
        return tuple(dict.fromkeys(violation.rule for violation in self.violations))

    def rows_for(self, rule: str) -> tuple[int, ...]:
        """Every row index the named rule objected to, in order."""
        rows = {row for violation in self.violations if violation.rule == rule
                for row in violation.rows}
        return tuple(sorted(rows))

    def summary(self) -> str:
        """The report as a human reads it in a failing pipeline's log."""
        if self.ok:
            return (
                f"PASS  {CONTRACT_NAME} data contract: {self.rows_checked} rows, "
                f"{len(RULE_NAMES)} rules, no violations."
            )
        head = (
            f"FAIL  {CONTRACT_NAME} data contract: {len(self.failed_rules)} of "
            f"{len(RULE_NAMES)} rules violated over {self.rows_checked} rows."
        )
        return "\n\n".join([head, *(str(violation) for violation in self.violations)])


def _rule_name(check: object) -> str:
    """Translate one Pandera check identifier into one of this module's rules."""
    name = str(check)
    if name in RULES:
        return name
    if name.startswith("dtype("):
        return RULE_COLUMN_TYPES
    return _BUILT_IN_RULES.get(name, name)


def _rows(indices: pd.Series) -> tuple[int, ...]:
    """The row indices in a group of failure cases, deduplicated and ordered.

    Schema-level failures - a missing column, a wrong row count - point at no
    particular row, and come back as an empty tuple rather than a fabricated one.
    """
    usable = pd.to_numeric(indices, errors="coerce").dropna()
    return tuple(sorted({int(index) for index in usable}))


#: The rules whose check asks a question about the frame, or about a column, as
#: a whole - "are there 398 rows?", "are there six holes?" - rather than about
#: each value in turn. Pandera reports those with the check's own boolean result
#: standing in for the offending value, which tells a reader nothing and which
#: pandas will happily render as ``0.0`` once another failure has promoted the
#: column's dtype. The report drops it; the rule's description is the message.
_WHOLE_FRAME_RULES = frozenset({RULE_ROW_COUNT, RULE_HORSEPOWER_MISSING_COUNT})


def _values(cases: pd.Series) -> tuple[str, ...]:
    """The distinct offending values in a group of failure cases."""
    return tuple(dict.fromkeys(str(case) for case in cases.tolist()))


#: Stands in for "no column" while grouping: pandas turns ``None`` into ``NaN``
#: in a group key, and ``NaN`` is not a column name either.
_NO_COLUMN = ""


def _violations(failure_cases: pd.DataFrame) -> tuple[RuleViolation, ...]:
    """Turn Pandera's failure-case table into this module's report."""
    annotated = failure_cases.assign(
        rule=[_rule_name(check) for check in failure_cases["check"]],
        # A column-scoped failure names its column; a schema-scoped one names
        # the schema, which is not a column and must not be reported as one.
        where=[
            column if context == "Column" else _NO_COLUMN
            for context, column in zip(failure_cases["schema_context"], failure_cases["column"])
        ],
    )
    violations = [
        RuleViolation(
            rule=rule,
            column=where or None,
            rows=_rows(group["index"]),
            values=() if rule in _WHOLE_FRAME_RULES else _values(group["failure_case"]),
        )
        for (rule, where), group in annotated.groupby(["rule", "where"], sort=True)
    ]
    return tuple(sorted(violations, key=lambda violation: (violation.rule, violation.column or "")))


def validate(frame: pd.DataFrame) -> ValidationReport:
    """Check ``frame`` against every rule at once and report what it broke.

    Never raises on invalid data and never modifies the frame: the caller
    decides what a violation means. The pipeline's validate step decides it
    means a non-zero exit code, which is what stops compute being spent on a
    bad run.
    """
    try:
        SCHEMA.validate(frame, lazy=True)
    except SchemaErrors as errors:
        return ValidationReport(
            rows_checked=len(frame),
            violations=_violations(errors.failure_cases),
        )
    return ValidationReport(rows_checked=len(frame))
