"""Regression tests for the reproducibility rule.

This repository's predecessor died of an unpinned dependency: nobody changed the
code, scikit-learn 1.2 removed a constructor parameter the training script
passed, and the training step stopped working. The rule that came out of that
incident - every direct dependency pinned to an exact version, in exactly three
manifests, with the versions aligned across them - is a correctness requirement,
so it is tested rather than merely written down.

A range specifier, a bare package name, or a fourth manifest fails the build.
"""

import re
from pathlib import Path

import pytest

ENVIRONMENTS = Path(__file__).resolve().parents[1] / "environments"

TRAINING = ENVIRONMENTS / "training.conda.yaml"
SERVING = ENVIRONMENTS / "serving.requirements.txt"
DEV = ENVIRONMENTS / "dev.requirements.txt"
LOCKFILE = ENVIRONMENTS / "dev.lock.txt"

MANIFESTS = (TRAINING, SERVING, DEV)

# name==version, optionally with extras and an environment marker.
EXACT_PIN = re.compile(r"^[A-Za-z0-9._-]+(\[[A-Za-z0-9,._-]+\])?==[^,<>!~ ]+( ;.*)?$")


def requirement_lines(path: Path) -> list[str]:
    """Every dependency line in a requirements file, comments and blanks dropped."""
    return [
        line.strip()
        for line in path.read_text().splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]


def conda_dependency_lines(path: Path) -> list[str]:
    """Every dependency in a conda environment file, from both the conda and pip sections."""
    lines = []
    in_dependencies = False
    for raw in path.read_text().splitlines():
        stripped = raw.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if not raw[0].isspace() and not stripped.startswith("- "):
            in_dependencies = stripped.startswith("dependencies:")
            continue
        if not in_dependencies or not stripped.startswith("- ") or stripped == "- pip:":
            continue
        lines.append(stripped[2:])
    return lines


def package_name(requirement: str) -> str:
    return re.split(r"[=<>!~ \[]", requirement, maxsplit=1)[0].lower().replace("_", "-")


def pinned_version(requirement: str) -> str:
    return re.split(r"[=]+", requirement.split(";")[0].strip(), maxsplit=1)[-1].strip()


def test_exactly_three_manifests_exist():
    """Three manifests, each with one job. A fourth means a job went undocumented."""
    present = sorted(p.name for p in ENVIRONMENTS.iterdir() if p.is_file())

    assert present == sorted(
        [TRAINING.name, SERVING.name, DEV.name, LOCKFILE.name]
    ), f"unexpected file in environments/: {present}"


@pytest.mark.parametrize("manifest", MANIFESTS, ids=lambda p: p.name)
def test_every_manifest_documents_its_job(manifest):
    assert manifest.read_text().lstrip().startswith("#"), (
        f"{manifest.name} must open with a comment saying which of the three jobs it does"
    )


@pytest.mark.parametrize("manifest", (SERVING, DEV), ids=lambda p: p.name)
def test_requirements_are_pinned_to_an_exact_version(manifest):
    unpinned = [line for line in requirement_lines(manifest) if not EXACT_PIN.match(line)]

    assert not unpinned, f"{manifest.name} has requirements that are not pinned: {unpinned}"


def test_conda_dependencies_are_pinned_to_an_exact_version():
    loose = [
        dependency
        for dependency in conda_dependency_lines(TRAINING)
        if not re.match(r"^[A-Za-z0-9._-]+={1,2}[0-9][^,<>!~*]*$", dependency)
    ]

    assert not loose, f"{TRAINING.name} has dependencies that are not pinned: {loose}"


def test_lockfile_pins_every_transitive_dependency():
    unpinned = [line for line in requirement_lines(LOCKFILE) if not EXACT_PIN.match(line)]

    assert not unpinned, f"{LOCKFILE.name} has entries that are not pinned: {unpinned}"


def test_lockfile_covers_every_direct_development_dependency():
    locked = {package_name(line) for line in requirement_lines(LOCKFILE)}
    direct = {package_name(line) for line in requirement_lines(DEV)}

    assert direct <= locked, f"missing from the lockfile: {sorted(direct - locked)}"


def test_shared_packages_use_the_same_version_everywhere():
    """CI must test the code against the versions training and serving will run."""
    versions: dict[str, dict[str, str]] = {}
    sources = {
        TRAINING.name: conda_dependency_lines(TRAINING),
        SERVING.name: requirement_lines(SERVING),
        DEV.name: requirement_lines(DEV),
    }
    for source, lines in sources.items():
        for line in lines:
            versions.setdefault(package_name(line), {})[source] = pinned_version(line)

    disagreements = {
        package: seen
        for package, seen in versions.items()
        if len(seen) > 1 and len(set(seen.values())) > 1
    }

    assert not disagreements, f"the same package is pinned differently: {disagreements}"


def test_python_is_pinned_to_311():
    """3.11, not 3.12+: the widest overlap with the platform's curated images."""
    python = [d for d in conda_dependency_lines(TRAINING) if package_name(d) == "python"]

    assert python, "the training environment must pin the interpreter"
    assert pinned_version(python[0]).startswith("3.11.")
