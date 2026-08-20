"""Step 4 of the training pipeline: register the promoted model.

Run as ``automobile-register`` or ``python -m automobile.entrypoints.register``.

The step runs only when the quality gate promoted the candidate, and it records
the version of the data asset the model was trained on so that a model's
lineage identifies exactly which data produced it.

The registration body arrives with the quality-gate slice; the command-line
surface is established here.
"""

import argparse
import sys

_PENDING = (
    "The register step is scaffolding until the quality-gate slice fills it in. "
    "See the 'Quality gate, evaluate and register entrypoints' issue."
)


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser for the register step."""
    return argparse.ArgumentParser(
        prog="automobile-register",
        description="Register the promoted model against the data asset it was trained on.",
    )


def main(argv: list[str] | None = None) -> int:
    """Parse arguments and run the register step."""
    build_parser().parse_args(argv)
    raise NotImplementedError(_PENDING)


if __name__ == "__main__":  # pragma: no cover - exercised as a subprocess
    sys.exit(main())
