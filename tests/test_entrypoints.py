"""Behaviour tests for the command-line skeleton.

These assert on what a caller can observe from outside: that each pipeline step
is reachable as a module and as an installed console script, that asking it for
help succeeds, and that it refuses arguments it does not understand instead of
silently ignoring them. They say nothing about how any step is implemented, so
they survive the slices that fill the steps in.

They also make the harness itself falsifiable. The predecessor's only test was
never executed, because the CI template that would have run it was commented
out; a test suite that cannot fail is indistinguishable from no test suite.
"""

import subprocess
import sys
from importlib import metadata

import pytest

from automobile.entrypoints import STEPS


def run_step(step: str, *args: str) -> subprocess.CompletedProcess:
    """Run one pipeline step in a subprocess, as a user would from a shell."""
    return subprocess.run(
        [sys.executable, "-m", f"automobile.entrypoints.{step}", *args],
        capture_output=True,
        text=True,
    )


def test_all_four_pipeline_steps_are_declared():
    assert STEPS == ("validate", "train", "evaluate", "register")


@pytest.mark.parametrize("step", STEPS)
def test_step_help_succeeds(step):
    result = run_step(step, "--help")

    assert result.returncode == 0, result.stderr
    assert "usage" in result.stdout.lower()
    assert f"automobile-{step}" in result.stdout


@pytest.mark.parametrize("step", STEPS)
def test_step_rejects_an_argument_it_does_not_understand(step):
    result = run_step(step, "--not-a-real-option")

    assert result.returncode != 0
    assert result.stderr.strip(), "a rejected argument must say why"


@pytest.mark.parametrize("step", STEPS)
def test_step_is_installed_as_a_console_script(step):
    scripts = {
        entry.name: entry
        for entry in metadata.entry_points(group="console_scripts")
        if entry.name.startswith("automobile-")
    }

    assert f"automobile-{step}" in scripts, (
        "the package must be installed for its console scripts to resolve: "
        "run `pip install -e .` (see the README)"
    )
    assert callable(scripts[f"automobile-{step}"].load())
