"""The dataset's schema, and the one function that reads it off disk.

The data is the canonical UCI "Auto MPG" set: 398 cars, nine columns, and six
rows whose ``horsepower`` is the string ``?`` rather than a number. It is
committed to this repository as a seed fixture (``data/auto-mpg.csv``) rather
than downloaded at run time, because determinism in a classroom outweighs
purity: an upstream host that is unreachable at the start of a session is not a
risk worth taking.

The six sentinel values are deliberately preserved. They are the dataset's real
defect, they are what the data contract is written against, and they are what
the model pipeline has to survive without any help from its caller. A
pre-cleaned copy of this file would delete the only genuine data-quality problem
the course has.

The names below exist so that no other module has to spell a column name as a
string literal. Changing the dataset means changing this module and the data
contract - not hunting through the training script, which is how the
predecessor's column handling ended up scattered across several files.
"""

from pathlib import Path

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


def load_dataset(path: Path | str | None = None) -> pd.DataFrame:
    """Read the dataset from ``path``, or from the committed seed fixture.

    Nothing is cleaned, coerced or dropped here. The frame comes back exactly as
    the file has it - sentinel values included - because validating it is the
    data contract's job and repairing it is the model pipeline's.
    """
    return pd.read_csv(Path(path) if path is not None else SEED_DATA_PATH)
