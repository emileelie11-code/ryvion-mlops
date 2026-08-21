"""The dataset's schema, and the one function that reads it off disk.

The data is the canonical UCI "Auto MPG" set: 398 cars, nine columns, and six
rows whose ``horsepower`` is the string ``?`` rather than a number. It is
committed to this repository as a seed fixture (``data/auto-mpg.csv``) rather
than downloaded at run time, because determinism in a classroom outweighs
purity: an upstream host that is unreachable at the start of a session is not a
risk worth taking.

The six sentinel values are deliberately preserved *in the file*. They are the
dataset's real defect, they are what the data contract is written against, and a
pre-cleaned copy of the CSV would delete the only genuine data-quality problem
the course has.

They do not survive the loader. :func:`parse_sentinels` turns ``?`` into a
proper missing value on the way in, and :func:`load_dataset` applies it. That
line - the loader - is the data boundary, and the split of responsibilities
across it is the point:

*Stateless format parsing happens here.* ``?`` is a 1983 fixed-width-file
encoding artifact, not a domain concept. Converting it fits nothing, learns
nothing and depends on no other row, so doing it at the boundary is free of
consequence - and it buys a model whose declared input schema says ``horsepower``
is a *double* that may be ``null``, rather than a *string* whose callers have to
know to send ``"130.0"`` and to spell "unknown" as a question mark.

*Fitted, data-dependent transforms do not.* The mean that fills those missing
values is learned from the training set, and it stays inside the model pipeline
where it is fitted, serialised and shipped along with the estimator. Moving
*that* out here is exactly the training/serving skew this repository was rebuilt
to close, so it does not move.

The rules that check the result live in :mod:`automobile.data_contract`, which
reads this module rather than the other way round - the dependency runs one way,
so there is one place a column name is spelled.

The names below exist so that no other module has to spell a column name as a
string literal. Changing the dataset means changing this module and the data
contract - not hunting through the training script, which is how the
predecessor's column handling ended up scattered across several files.
"""

from pathlib import Path

import numpy as np
import pandas as pd

#: The committed seed fixture. Cloud jobs are handed a versioned data asset
#: instead; this path is the local default, not a hard-coded dependency.
SEED_DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "auto-mpg.csv"

#: Every column, in the order the file carries them.
COLUMNS = (
    "mpg",
    "cylinders",
    "displacement",
    "horsepower",
    "weight",
    "acceleration",
    "model year",
    "origin",
    "car name",
)

#: What the model predicts.
TARGET = "mpg"

#: Everything the model is given - including the free-text column, which the
#: pipeline drops itself so that callers may pass a row exactly as it appears in
#: the data.
FEATURES = tuple(column for column in COLUMNS if column != TARGET)

#: The free-text column. It identifies a car; it must never reach the estimator.
FREE_TEXT_FEATURE = "car name"

#: The columns the model actually learns from.
NUMERIC_FEATURES = tuple(column for column in FEATURES if column != FREE_TEXT_FEATURE)

#: The non-numeric value that stands in for a missing measurement, and the one
#: column that carries it.
SENTINEL = "?"
SENTINEL_COLUMN = "horsepower"


def parse_sentinels(frame: pd.DataFrame) -> pd.DataFrame:
    """Return ``frame`` with the ``?`` sentinel read as a missing value.

    The only transformation this repository performs outside the model, and it
    is deliberately the most boring one available: every cell of
    ``horsepower`` holding the exact string ``?`` becomes ``NaN``, and the
    column is then read as a number. Nothing is imputed, nothing is dropped,
    nothing is learned from the data, and calling this twice does the same as
    calling it once.

    What it does *not* do is hide anything from the data contract. A value that
    is neither the known sentinel nor a number - a stray ``n/a``, a decimal
    comma - is left exactly where it is, so the column stays non-numeric and
    :func:`automobile.data_contract.validate` reports it against the row it is
    in. Silently coercing unknown junk to "missing" would turn a data-quality
    failure into a quietly worse model.
    """
    if SENTINEL_COLUMN not in frame.columns:
        return frame

    parsed = frame.copy()
    column = parsed[SENTINEL_COLUMN]
    if column.dtype != object:
        return parsed

    without_sentinels = column.mask(column == SENTINEL, np.nan)
    try:
        parsed[SENTINEL_COLUMN] = pd.to_numeric(without_sentinels, errors="raise")
    except (TypeError, ValueError):
        parsed[SENTINEL_COLUMN] = without_sentinels
    return parsed


def load_dataset(path: Path | str | None = None) -> pd.DataFrame:
    """Read the dataset from ``path``, or from the committed seed fixture.

    The only thing done to the file's contents is :func:`parse_sentinels` - the
    stateless boundary parse described in this module's docstring. Nothing is
    imputed, nothing is dropped and no column is renamed: checking the frame is
    the data contract's job and repairing it is the model pipeline's.
    """
    return parse_sentinels(pd.read_csv(Path(path) if path is not None else SEED_DATA_PATH))
