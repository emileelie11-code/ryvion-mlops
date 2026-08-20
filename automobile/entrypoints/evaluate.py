"""Step 3 of the training pipeline: promote-or-reject the candidate model.

Run as ``automobile-evaluate`` or ``python -m automobile.entrypoints.evaluate``.

The decision itself is a pure function in the quality-gate module, so that the
threshold policy can be changed, unit tested and deliberately broken in class
without touching any platform integration. This shell turns the decision into
an exit code: a rejected candidate exits non-zero, the job fails, and the
register step never runs.

That exit code replaces the predecessor's trick of cancelling its own parent
run. The run now reports as failed rather than cancelled, which is both simpler
and more honest.

The gate and this shell's body arrive with the quality-gate slice; the
command-line surface is established here.
"""

import argparse
import sys

_PENDING = (
    "The evaluate step is scaffolding until the quality-gate slice fills it in. "
    "See the 'Quality gate, evaluate and register entrypoints' issue."
)


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser for the evaluate step."""
    return argparse.ArgumentParser(
        prog="automobile-evaluate",
        description="Compare the candidate model against the incumbent and apply the gate.",
    )


def main(argv: list[str] | None = None) -> int:
    """Parse arguments and run the evaluate step."""
    build_parser().parse_args(argv)
    raise NotImplementedError(_PENDING)


if __name__ == "__main__":  # pragma: no cover - exercised as a subprocess
    sys.exit(main())
