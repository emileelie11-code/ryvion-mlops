"""Step 1 of the training pipeline: check the data against the contract.

Run as ``automobile-validate`` or ``python -m automobile.entrypoints.validate``.

The rules themselves live in :mod:`automobile.data_contract`, behind a single
``validate(frame)`` call, so that changing the dataset never means hunting
through the training script. This shell only reads the frame, calls the
contract and turns the report into an exit code: a failing contract stops the
pipeline before any compute is spent on training.

The frame it checks is the one training will see - read through
:func:`automobile.dataset.load_dataset`, so the ``?`` sentinel has already
become a missing value. That is deliberate. Validating some other, rawer frame
would be validating something no model ever consumes.

It prints the whole report rather than the first failure, because the point of a
gate is to tell a student what is wrong with their data in one run rather than in
six.
"""

import argparse
import sys
from pathlib import Path

from automobile import dataset
from automobile.data_contract import validate as check_contract

#: Exit codes. Zero is the only one the pipeline lets through to training.
CONTRACT_HONOURED = 0
CONTRACT_VIOLATED = 1
DATA_UNREADABLE = 2


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser for the validate step."""
    parser = argparse.ArgumentParser(
        prog="automobile-validate",
        description="Check the automobile dataset against its data contract.",
        epilog=(
            "Exits 0 when every rule holds, 1 when the data violates the contract, "
            "and 2 when the data cannot be read at all. The pipeline runs this as "
            "step zero so that a bad dataset costs nothing but this command."
        ),
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=dataset.SEED_DATA_PATH,
        help="CSV to check (default: the committed seed dataset).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Parse arguments and run the validate step."""
    args = build_parser().parse_args(argv)

    try:
        frame = dataset.load_dataset(args.data)
    except (OSError, ValueError) as unreadable:
        print(f"FAIL  {args.data} could not be read as the automobile dataset: {unreadable}")
        return DATA_UNREADABLE

    report = check_contract(frame)
    print(f"data: {args.data}")
    print(report.summary())
    return CONTRACT_HONOURED if report.ok else CONTRACT_VIOLATED


if __name__ == "__main__":  # pragma: no cover - exercised as a subprocess
    sys.exit(main())
