"""The promote-or-reject decision, as a pure function.

In the repository this replaces, the decision was entangled with the platform:
the evaluation script compared two numbers and then reached up to cancel its own
parent pipeline run. The policy could not be read without reading the SDK calls
around it, could not be unit tested without a workspace, and could not be
changed without touching platform integration.

Here it is a function of three arguments and nothing else. It reads no files,
imports no tracking library and knows nothing about where the numbers came from
- which is why the module imports only the standard library, and why a test
asserts that it still does. The entrypoint next door fetches the metrics and
turns the answer into an exit code; that part is I/O and lives at the boundary.

Two behaviours are worth stating out loud, because they are the ones that bite:

**No incumbent promotes.** The first model ever trained has nothing to beat, and
a gate that refused it would deadlock the pipeline on its first run. An empty
mapping of incumbent metrics means the same thing as ``None``: nothing has been
registered yet.

**The threshold is a policy argument, not a constant in this file.** The default
below is a default, not a rule. Changing the metric, the direction of "better",
or the margin a candidate has to clear is a change of configuration - which is
what makes the gate something an instructor can deliberately break in front of a
class, and put back, without editing any code that talks to a platform.
"""

import math
from enum import StrEnum
from typing import Mapping, NamedTuple


class Goal(StrEnum):
    """Which direction counts as better for the metric being gated on."""

    MINIMISE = "minimise"
    MAXIMISE = "maximise"


class Reason(StrEnum):
    """Why the gate decided what it decided.

    A code rather than a sentence, so that a caller - or a test - can assert on
    the reason without depending on the wording of a log line.
    """

    NO_INCUMBENT = "no-incumbent"
    MEETS_THRESHOLD = "meets-threshold"
    BELOW_THRESHOLD = "below-threshold"
    NOT_A_NUMBER = "not-a-number"


class ThresholdPolicy(NamedTuple):
    """What "good enough" means, in a form the operator can change.

    ``metric``
        the key to compare, as it is named in the metrics mapping.
    ``goal``
        whether a smaller or a larger value is better. Error measures minimise;
        scores like ``r2`` maximise.
    ``min_improvement``
        how much better than the incumbent the candidate has to be, as an
        absolute margin in the metric's own units. Zero - the default - means a
        candidate that merely matches the incumbent is promoted. A negative
        margin would mean "register a model known to be worse", which this gate
        declines to express; :func:`decide` rejects it.
    """

    metric: str
    goal: Goal = Goal.MINIMISE
    min_improvement: float = 0.0


#: The policy applied when the operator names none. Mean squared error on the
#: held-out half, smaller being better, with no margin demanded. It is a
#: starting point for a course to argue with, not a recommendation.
DEFAULT_POLICY = ThresholdPolicy(metric="test_mse", goal=Goal.MINIMISE, min_improvement=0.0)


class GateDecision(NamedTuple):
    """The verdict, and enough context to explain it in a log or a run tag."""

    promote: bool
    reason: Reason
    metric: str
    candidate: float
    #: The incumbent's value, or ``None`` when there is no incumbent.
    incumbent: float | None
    #: The value the candidate had to reach, or ``None`` when nothing was to beat.
    required: float | None

    def summary(self) -> str:
        """One line, for a human reading the job log."""
        verdict = "promote" if self.promote else "reject"
        if self.reason is Reason.NOT_A_NUMBER:
            return f"{verdict}: {self.metric} is not a number ({self.candidate})"
        if self.reason is Reason.NO_INCUMBENT:
            return (
                f"{verdict}: no incumbent to beat, so {self.metric}={self.candidate} "
                "becomes the baseline"
            )
        comparison = "clears" if self.promote else "misses"
        return (
            f"{verdict}: {self.metric}={self.candidate} {comparison} the required "
            f"{self.required} (incumbent {self.incumbent})"
        )


def required_value(incumbent: float, policy: ThresholdPolicy) -> float:
    """The value a candidate must reach to beat ``incumbent`` under ``policy``.

    Exposed rather than hidden because the number is worth printing: "you needed
    11.4 and scored 11.9" is a more useful failure than "rejected".
    """
    if policy.goal is Goal.MINIMISE:
        return incumbent - policy.min_improvement
    return incumbent + policy.min_improvement


def decide(
    candidate: Mapping[str, float],
    incumbent: Mapping[str, float] | None,
    policy: ThresholdPolicy = DEFAULT_POLICY,
) -> GateDecision:
    """Promote or reject ``candidate``, given ``incumbent`` and ``policy``.

    ``incumbent`` is ``None`` - or an empty mapping - when nothing has been
    registered yet, in which case the candidate is promoted: the first model
    through has nothing to beat.

    The comparison is inclusive at the boundary. A candidate that lands exactly
    on the required value is promoted, so ``min_improvement`` reads as "must be
    at least this much better" rather than "must be more than this much better".

    Raises ``ValueError`` if the metric named by the policy is missing from
    either mapping, or if the policy demands a negative improvement. Both are
    operator mistakes, and both are worse when they are guessed at: a missing
    metric silently treated as absent would promote every candidate.
    """
    if policy.min_improvement < 0:
        raise ValueError(
            f"min_improvement must not be negative, got {policy.min_improvement}: "
            "a negative margin would promote a model known to be worse"
        )

    candidate_value = _read(candidate, policy.metric, "candidate")

    if math.isnan(candidate_value):
        return GateDecision(
            promote=False,
            reason=Reason.NOT_A_NUMBER,
            metric=policy.metric,
            candidate=candidate_value,
            incumbent=None,
            required=None,
        )

    if not incumbent:
        return GateDecision(
            promote=True,
            reason=Reason.NO_INCUMBENT,
            metric=policy.metric,
            candidate=candidate_value,
            incumbent=None,
            required=None,
        )

    incumbent_value = _read(incumbent, policy.metric, "incumbent")
    required = required_value(incumbent_value, policy)
    if policy.goal is Goal.MINIMISE:
        promote = candidate_value <= required
    else:
        promote = candidate_value >= required

    return GateDecision(
        promote=promote,
        reason=Reason.MEETS_THRESHOLD if promote else Reason.BELOW_THRESHOLD,
        metric=policy.metric,
        candidate=candidate_value,
        incumbent=incumbent_value,
        required=required,
    )


def _read(metrics: Mapping[str, float], name: str, whose: str) -> float:
    """Pull one metric out of a mapping, or say precisely what was missing."""
    try:
        return float(metrics[name])
    except KeyError:
        available = ", ".join(sorted(metrics)) or "none at all"
        raise ValueError(
            f"the {whose} has no metric named {name!r}; it has: {available}"
        ) from None
