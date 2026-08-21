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

Belt and braces, because an exit code is only as good as the thing reading it:
the verdict is also written onto the candidate run as a tag, and the register
step refuses to register anything that is not tagged as promoted. Running the
two commands by hand, in the wrong order, on purpose, still cannot register a
rejected model.

The registry this step reads the incumbent from needs a database behind it, so
locally the tracking destination is a SQLite file rather than the ``./mlruns``
directory the train step is happy with. That is one environment variable, and
the README spells it out.
"""

import argparse
import sys

from automobile.quality_gate import DEFAULT_POLICY, Goal, ThresholdPolicy, decide
from automobile.tracking import (
    DEFAULT_REGISTERED_MODEL_NAME,
    find_incumbent,
    record_gate_decision,
    run_metrics,
)


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser for the evaluate step."""
    parser = argparse.ArgumentParser(
        prog="automobile-evaluate",
        description="Compare the candidate model against the incumbent and apply the gate.",
        epilog=(
            "Exits 0 when the candidate is promoted and 1 when it is rejected, so that "
            "a pipeline stops before the register step. The threshold policy is these "
            "three flags: nothing about it is compiled into the gate. Reading the "
            "incumbent needs a registry, which needs a database-backed tracking "
            "destination: locally, MLFLOW_TRACKING_URI=sqlite:///mlflow.db."
        ),
    )
    parser.add_argument(
        "--run-id",
        required=True,
        help="The training run holding the candidate's metrics (printed by the train step).",
    )
    parser.add_argument(
        "--model-name",
        default=DEFAULT_REGISTERED_MODEL_NAME,
        help="Registered model whose latest version is the incumbent (default: %(default)s).",
    )
    parser.add_argument(
        "--metric",
        default=DEFAULT_POLICY.metric,
        help="Metric the gate decides on (default: %(default)s).",
    )
    parser.add_argument(
        "--goal",
        default=DEFAULT_POLICY.goal.value,
        choices=[goal.value for goal in Goal],
        help="Whether a smaller or a larger value is better (default: %(default)s).",
    )
    parser.add_argument(
        "--min-improvement",
        type=float,
        default=DEFAULT_POLICY.min_improvement,
        help=(
            "How much better than the incumbent the candidate must be, in the metric's "
            "own units (default: %(default)s)."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Parse arguments and run the evaluate step."""
    args = build_parser().parse_args(argv)

    policy = ThresholdPolicy(
        metric=args.metric,
        goal=Goal(args.goal),
        min_improvement=args.min_improvement,
    )

    candidate = run_metrics(args.run_id)
    incumbent = find_incumbent(args.model_name)
    decision = decide(candidate, incumbent.metrics if incumbent else None, policy)
    verdict = record_gate_decision(
        args.run_id,
        promote=decision.promote,
        reason=str(decision.reason),
        metric=decision.metric,
    )

    print(f"candidate run:   {args.run_id}")
    print(f"registered as:   {args.model_name}")
    if incumbent is None:
        print("incumbent:       none registered yet")
    else:
        print(f"incumbent:       version {incumbent.version} (run {incumbent.run_id})")
    print(f"policy:          {policy.metric} {policy.goal}, margin {policy.min_improvement}")
    print(f"decision:        {verdict} ({decision.reason})")
    print(f"why:             {decision.summary()}")

    return 0 if decision.promote else 1


if __name__ == "__main__":  # pragma: no cover - exercised as a subprocess
    sys.exit(main())
