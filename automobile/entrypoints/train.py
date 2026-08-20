"""Step 2 of the training pipeline: fit the model and log the artifact.

Run as ``automobile-train`` or ``python -m automobile.entrypoints.train``.

The estimator is obtained from the model factory rather than constructed here,
so that training, any future retraining and the serving container cannot drift
apart. The artifact logged is a complete pipeline - preprocessing, scaling and
estimator together - carrying its own fitted preprocessing statistics, a
signature and an input example.

The tracking destination is read from the environment and never hard-coded: with
nothing configured the run lands in a local file-based store and needs no cloud
account at all.

The training body arrives with the local training slice; the command-line
surface is established here.
"""

import argparse
import sys

_PENDING = (
    "The train step is scaffolding until the local training slice fills it in. "
    "See the 'Local training tracer bullet' issue."
)


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser for the train step."""
    return argparse.ArgumentParser(
        prog="automobile-train",
        description="Fit the automobile model pipeline and log it to the tracking store.",
    )


def main(argv: list[str] | None = None) -> int:
    """Parse arguments and run the train step."""
    build_parser().parse_args(argv)
    raise NotImplementedError(_PENDING)


if __name__ == "__main__":  # pragma: no cover - exercised as a subprocess
    sys.exit(main())
