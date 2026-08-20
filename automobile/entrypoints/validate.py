"""Step 1 of the training pipeline: check the data against the contract.

Run as ``automobile-validate`` or ``python -m automobile.entrypoints.validate``.

The rules themselves live in the data-contract module, behind a single
``validate(frame)`` call, so that changing the dataset never means hunting
through the training script. This shell only reads the frame, calls the
contract and turns the report into an exit code: a failing contract stops the
pipeline before any compute is spent on training.

The contract and this shell's body arrive with the data-contract slice; the
command-line surface is established here.
"""

import argparse
import sys

_PENDING = (
    "The validate step is scaffolding until the data-contract slice fills it in. "
    "See the 'Data contract and validate entrypoint' issue."
)


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser for the validate step."""
    return argparse.ArgumentParser(
        prog="automobile-validate",
        description="Check the automobile dataset against its data contract.",
    )


def main(argv: list[str] | None = None) -> int:
    """Parse arguments and run the validate step."""
    build_parser().parse_args(argv)
    raise NotImplementedError(_PENDING)


if __name__ == "__main__":  # pragma: no cover - exercised as a subprocess
    sys.exit(main())
