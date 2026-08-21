"""Behaviour tests for the promote-or-reject gate.

The gate is the module an instructor breaks in front of a class and puts back,
so these tests are written to say what it *does*, not how it does it. Nothing
here names a number the model actually produces: the metrics are invented, the
assertions are about the relationships between them, and every threshold case is
expressed as "one step better" or "one step worse" rather than as a value
someone once observed.

The boundary cases use ``math.nextafter``, which is the smallest amount a float
can move. A gate that got its inclusive comparison backwards passes a test built
on values a whole unit apart and fails this one.
"""

import ast
import math
import sys
from pathlib import Path

import pytest

from automobile.quality_gate import (
    DEFAULT_POLICY,
    GateDecision,
    Goal,
    Reason,
    ThresholdPolicy,
    decide,
    required_value,
)

GATE_SOURCE = Path(__file__).resolve().parents[1] / "automobile" / "quality_gate.py"

#: A metric where smaller is better, and one where larger is better.
LOWER_IS_BETTER = ThresholdPolicy(metric="test_mse", goal=Goal.MINIMISE)
HIGHER_IS_BETTER = ThresholdPolicy(metric="test_r2", goal=Goal.MAXIMISE)


def nudge_up(value: float) -> float:
    """The next float above ``value`` - the smallest step there is."""
    return math.nextafter(value, math.inf)


def nudge_down(value: float) -> float:
    """The next float below ``value``."""
    return math.nextafter(value, -math.inf)


# --------------------------------------------------------------------------
# Promote, reject, and the case with nothing to beat
# --------------------------------------------------------------------------


def test_a_candidate_that_improves_on_the_incumbent_is_promoted():
    decision = decide({"test_mse": 8.0}, {"test_mse": 10.0}, LOWER_IS_BETTER)

    assert decision.promote
    assert decision.reason is Reason.MEETS_THRESHOLD


def test_a_candidate_that_is_worse_than_the_incumbent_is_rejected():
    decision = decide({"test_mse": 12.0}, {"test_mse": 10.0}, LOWER_IS_BETTER)

    assert not decision.promote
    assert decision.reason is Reason.BELOW_THRESHOLD


def test_the_first_model_ever_trained_is_promoted():
    decision = decide({"test_mse": 10.0}, None, LOWER_IS_BETTER)

    assert decision.promote
    assert decision.reason is Reason.NO_INCUMBENT
    assert decision.incumbent is None


def test_an_empty_incumbent_means_the_same_as_no_incumbent():
    """Whether "nothing registered" arrives as None or as {} is the caller's detail."""
    assert decide({"test_mse": 10.0}, {}, LOWER_IS_BETTER) == decide(
        {"test_mse": 10.0}, None, LOWER_IS_BETTER
    )


def test_a_terrible_first_model_is_still_promoted():
    """No incumbent means no bar: a gate that refused would deadlock run one."""
    decision = decide({"test_mse": 1e9}, None, LOWER_IS_BETTER)

    assert decision.promote


def test_the_direction_of_better_decides_the_verdict():
    """The same two numbers, opposite goals, opposite answers."""
    candidate, incumbent = {"m": 0.9}, {"m": 0.5}

    assert decide(candidate, incumbent, ThresholdPolicy("m", Goal.MAXIMISE)).promote
    assert not decide(candidate, incumbent, ThresholdPolicy("m", Goal.MINIMISE)).promote


def test_a_higher_is_better_metric_rejects_a_lower_candidate():
    decision = decide({"test_r2": 0.4}, {"test_r2": 0.8}, HIGHER_IS_BETTER)

    assert not decision.promote
    assert decision.reason is Reason.BELOW_THRESHOLD


# --------------------------------------------------------------------------
# The threshold boundary
# --------------------------------------------------------------------------


def test_matching_the_incumbent_exactly_is_promoted_when_no_margin_is_demanded():
    decision = decide({"test_mse": 10.0}, {"test_mse": 10.0}, LOWER_IS_BETTER)

    assert decision.promote, "with a zero margin the comparison is inclusive"


def test_the_smallest_possible_regression_is_rejected():
    decision = decide({"test_mse": nudge_up(10.0)}, {"test_mse": 10.0}, LOWER_IS_BETTER)

    assert not decision.promote


def test_the_smallest_possible_improvement_is_promoted():
    decision = decide({"test_mse": nudge_down(10.0)}, {"test_mse": 10.0}, LOWER_IS_BETTER)

    assert decision.promote


@pytest.mark.parametrize("goal", list(Goal))
def test_landing_exactly_on_the_required_value_is_promoted(goal):
    """The documented boundary: "at least this much better", not "more than"."""
    policy = ThresholdPolicy(metric="m", goal=goal, min_improvement=1.5)
    required = required_value(10.0, policy)

    decision = decide({"m": required}, {"m": 10.0}, policy)

    assert decision.promote
    assert decision.required == required


@pytest.mark.parametrize("goal", list(Goal))
def test_falling_one_step_short_of_the_required_value_is_rejected(goal):
    policy = ThresholdPolicy(metric="m", goal=goal, min_improvement=1.5)
    required = required_value(10.0, policy)
    short = nudge_up(required) if goal is Goal.MINIMISE else nudge_down(required)

    assert not decide({"m": short}, {"m": 10.0}, policy).promote


def test_a_margin_rejects_an_improvement_too_small_to_clear_it():
    """The same candidate, promoted under one policy and rejected under a stricter one."""
    candidate, incumbent = {"test_mse": 9.5}, {"test_mse": 10.0}

    assert decide(candidate, incumbent, ThresholdPolicy("test_mse")).promote
    assert not decide(
        candidate, incumbent, ThresholdPolicy("test_mse", min_improvement=1.0)
    ).promote


def test_a_margin_never_makes_the_gate_easier_to_pass():
    incumbent = {"test_mse": 10.0}
    candidate = {"test_mse": 9.0}
    verdicts = [
        decide(candidate, incumbent, ThresholdPolicy("test_mse", min_improvement=margin)).promote
        for margin in (0.0, 0.5, 1.0, 2.0)
    ]

    assert verdicts == sorted(verdicts, reverse=True), "raising the bar cannot promote more"


# --------------------------------------------------------------------------
# The policy is configuration, not a constant in the module
# --------------------------------------------------------------------------


def test_the_metric_the_gate_decides_on_is_chosen_by_the_policy():
    """Two metrics disagreeing; whichever the policy names is the one that counts."""
    candidate = {"test_mse": 8.0, "test_mae": 5.0}
    incumbent = {"test_mse": 10.0, "test_mae": 2.0}

    assert decide(candidate, incumbent, ThresholdPolicy("test_mse")).promote
    assert not decide(candidate, incumbent, ThresholdPolicy("test_mae")).promote


def test_the_default_policy_is_a_default_rather_than_the_only_policy():
    candidate = {DEFAULT_POLICY.metric: 8.0, "custom": 1.0}
    incumbent = {DEFAULT_POLICY.metric: 10.0, "custom": 2.0}

    assert decide(candidate, incumbent).metric == DEFAULT_POLICY.metric
    assert decide(candidate, incumbent, ThresholdPolicy("custom")).metric == "custom"


def test_a_policy_naming_a_metric_nobody_recorded_says_so():
    with pytest.raises(ValueError) as raised:
        decide({"test_mse": 8.0}, {"test_mse": 10.0}, ThresholdPolicy("test_rmsle"))

    message = str(raised.value)
    assert "test_rmsle" in message
    assert "test_mse" in message, "the message should list what was available"


def test_a_metric_missing_from_the_incumbent_is_an_error_rather_than_a_promotion():
    """Silently treating it as absent would promote every candidate forever."""
    with pytest.raises(ValueError):
        decide({"test_mse": 8.0}, {"something_else": 10.0}, LOWER_IS_BETTER)


def test_a_policy_that_would_accept_a_worse_model_is_refused():
    with pytest.raises(ValueError):
        decide({"m": 1.0}, {"m": 10.0}, ThresholdPolicy("m", min_improvement=-5.0))


def test_a_candidate_whose_metric_is_not_a_number_is_never_promoted():
    """Not even with no incumbent, which is the case that would otherwise wave it through."""
    decision = decide({"test_mse": float("nan")}, None, LOWER_IS_BETTER)

    assert not decision.promote
    assert decision.reason is Reason.NOT_A_NUMBER


# --------------------------------------------------------------------------
# Purity
# --------------------------------------------------------------------------


def test_the_same_arguments_always_give_the_same_decision():
    candidate, incumbent = {"test_mse": 9.0}, {"test_mse": 10.0}

    first = decide(candidate, incumbent, LOWER_IS_BETTER)
    decide({"test_r2": 0.9}, {"test_r2": 0.5}, HIGHER_IS_BETTER)
    decide({"test_mse": 99.0}, None, ThresholdPolicy("test_mse", min_improvement=7.0))
    second = decide(candidate, incumbent, LOWER_IS_BETTER)

    assert first == second


def test_deciding_does_not_touch_its_arguments():
    candidate = {"test_mse": 9.0, "test_r2": 0.8}
    incumbent = {"test_mse": 10.0, "test_r2": 0.7}

    decide(candidate, incumbent, LOWER_IS_BETTER)

    assert candidate == {"test_mse": 9.0, "test_r2": 0.8}
    assert incumbent == {"test_mse": 10.0, "test_r2": 0.7}


def test_the_verdict_carries_the_numbers_it_was_reached_from():
    """A decision nobody can audit is a decision nobody will trust."""
    decision = decide({"test_mse": 9.0}, {"test_mse": 10.0}, LOWER_IS_BETTER)

    assert isinstance(decision, GateDecision)
    assert decision.candidate == 9.0
    assert decision.incumbent == 10.0
    assert decision.required == 10.0
    assert decision.metric == "test_mse"


def test_the_summary_says_which_way_the_verdict_went():
    promoted = decide({"test_mse": 9.0}, {"test_mse": 10.0}, LOWER_IS_BETTER).summary()
    rejected = decide({"test_mse": 11.0}, {"test_mse": 10.0}, LOWER_IS_BETTER).summary()

    assert promoted.startswith("promote")
    assert rejected.startswith("reject")
    assert "test_mse" in promoted and "test_mse" in rejected


def test_the_gate_imports_nothing_but_the_standard_library():
    """No I/O, no tracking library, no cloud - enforced rather than promised.

    This is the property that lets the gate be unit tested in milliseconds on a
    laptop with no account, and it is exactly the property that erodes the first
    time somebody finds it convenient to read the incumbent's metrics from inside
    the decision.
    """
    tree = ast.parse(GATE_SOURCE.read_text(encoding="utf-8"), str(GATE_SOURCE))
    roots = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            roots.add(node.module.split(".")[0])

    assert roots, "the scan must actually find the imports it is checking"
    assert roots <= set(sys.stdlib_module_names), f"non-standard-library imports: {roots}"
