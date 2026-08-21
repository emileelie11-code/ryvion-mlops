"""The two refusals, driven end to end: a rejected model never reaches the registry.

This is the most important property in the repository, and until now the only
things guaranteeing it were a transcript of a manual run and a CI workflow. Both
are evidence that it held once, on someone else's machine. Neither is a test.

What is under test is not one function but a *sequence*: four command-line steps,
each in its own process, exchanging state only through a tracking store - which
is what makes the refusals survive the steps running on four different machines,
as they do in ``.github/workflows/pipeline.yml``. So the steps are run as
subprocesses, in order, stopping at the first non-zero exit, exactly as a
pipeline runner stops. Calling ``main()`` in-process would test the same Python
and a different system.

Two independent mechanisms enforce the property, and both are covered here
because either one alone is a single point of failure:

1. **The exit code.** ``evaluate`` exits non-zero on a rejection, so the
   pipeline stops and ``register`` is never reached.
2. **The verdict on the run.** ``register`` reads the tag the gate wrote onto
   the candidate and refuses anything that is not a promotion - including a run
   that was never evaluated at all. This is the mechanism that survives someone
   typing the commands by hand, in the wrong order, on purpose.

**These tests assert on exit codes and on the state of the registry, never on
wording.** Both steps report which store they consulted and what it held, and
that reporting has already been improved once; a test that pinned the sentences
would have broken for a change that made the tool better. The registry is the
thing that must not move, so the registry is what is measured: the versions
present before a refusal and the versions present after it.

Isolation, because a test that registers a model into a developer's own store
has caused the very confusion the gate exists to prevent: every step runs in a
temporary directory with ``MLFLOW_TRACKING_URI=sqlite:///mlflow.db`` resolved
against that directory, with every inherited ``MLFLOW_*`` variable stripped from
the environment first. Both halves of that claim are asserted rather than
assumed: :func:`test_nothing_was_written_outside_the_temporary_store` checks that
the store is where it should be, and :func:`the_checkout_is_left_alone` fails the
module on teardown if the checkout gained a store it did not have before.

Marked ``integration`` and excluded from the default ``pytest`` run, because it
trains three real models and takes tens of seconds rather than milliseconds. It
runs in CI as its own step. To run it here::

    python -m pytest -m integration
"""

import os
import subprocess
import sys
from contextlib import chdir, contextmanager
from pathlib import Path
from typing import Callable, Iterator, NamedTuple

import pytest
from mlflow.tracking import MlflowClient

pytestmark = pytest.mark.integration

#: This checkout, which is what the subprocesses must exercise - not whatever an
#: editable install elsewhere happens to point at. It goes on PYTHONPATH because
#: the subprocesses run with their working directory in the temporary store.
REPO_ROOT = Path(__file__).resolve().parents[1]

HEALTHY_DATA = REPO_ROOT / "data" / "auto-mpg.csv"
CORRUPT_DATA = REPO_ROOT / "data" / "auto-mpg-corrupt.csv"

#: The store's two halves, as the README and the CI pipeline name them.
DATABASE = "mlflow.db"
ARTIFACTS = "mlruns"

EXPERIMENT = "refusal-paths"
MODEL_NAME = "automobile-mpg"

#: A margin in mean-squared-error units that no rerun of the same data on the
#: same seed can clear, which is how a deliberate rejection is arranged without
#: inventing a worse dataset. The same trick the pipeline workflow offers behind
#: its "Run workflow" button.
UNCLEARABLE_MARGIN = "5"

BASELINE = "baseline"
CANDIDATE = "candidate"
UNEVALUATED = "never-evaluated"


class StepResult(NamedTuple):
    """One step, run as its own process, and what a pipeline runner would see."""

    step: str
    returncode: int
    stdout: str
    stderr: str

    def failed(self) -> bool:
        return self.returncode != 0

    def report(self) -> str:
        """Everything the step said, for a failing assertion to carry."""
        return f"\n--- {self.step} exited {self.returncode} ---\n{self.stdout}\n{self.stderr}"


class Store(NamedTuple):
    """A throwaway tracking store: one directory, one SQLite file, no account."""

    directory: Path

    @property
    def database(self) -> Path:
        return self.directory / DATABASE

    @property
    def uri(self) -> str:
        """The absolute form, for reading the store back from this process."""
        return f"sqlite:///{self.database.as_posix()}"


@contextmanager
def reading(store: Store) -> Iterator[MlflowClient]:
    """A client on ``store``, used from inside it.

    The working directory matters even for a read. Opening a database-backed
    store resolves its default artifact root against the *current* directory and
    creates it, so a client built while pytest sits in the checkout drops an
    empty ``mlruns/`` there - the exact litter this test exists not to leave. So
    the read happens from inside the temporary store, where that directory is
    already the right one.
    """
    with chdir(store.directory):
        yield MlflowClient(tracking_uri=store.uri)


def environment_for(store: Store) -> dict[str, str]:
    """The environment a step runs in: this store, this checkout, nothing inherited.

    Every ``MLFLOW_*`` variable the developer or the CI runner set is dropped
    rather than overridden, so a variable this test does not know about cannot
    redirect a registration into somebody's real store.
    """
    environment = {
        name: value for name, value in os.environ.items() if not name.upper().startswith("MLFLOW_")
    }
    # Relative on purpose: it is the exact line the README tells a student to
    # export, and it resolves against the working directory the step is given.
    environment["MLFLOW_TRACKING_URI"] = f"sqlite:///{DATABASE}"
    environment["MLFLOW_EXPERIMENT_NAME"] = EXPERIMENT
    environment["PYTHONPATH"] = str(REPO_ROOT)
    environment["PYTHONNOUSERSITE"] = "1"
    return environment


def run_step(store: Store, step: str, *arguments: str) -> StepResult:
    """Run one pipeline step as its own process, from inside the temporary store."""
    completed = subprocess.run(
        [sys.executable, "-m", f"automobile.entrypoints.{step}", *arguments],
        cwd=store.directory,
        env=environment_for(store),
        capture_output=True,
        text=True,
    )
    return StepResult(step, completed.returncode, completed.stdout, completed.stderr)


Plan = list[tuple[str, Callable[[], list[str]]]]


def drive(store: Store, plan: Plan) -> list[StepResult]:
    """Run a plan of steps in order, stopping at the first non-zero exit.

    This is the whole of what a pipeline runner does, and it is why the steps'
    exit codes are load-bearing: nothing else here decides that training does
    not follow a failed contract check.

    Each step's arguments are built when the step is reached rather than up
    front, which is how the run id the train step creates reaches the two steps
    that need it.
    """
    results: list[StepResult] = []
    for step, arguments in plan:
        results.append(run_step(store, step, *arguments()))
        if results[-1].failed():
            break
    return results


def steps_that_ran(results: list[StepResult]) -> list[str]:
    return [result.step for result in results]


def report(results: list[StepResult]) -> str:
    return "".join(result.report() for result in results)


def run_id_named(store: Store, run_name: str) -> str:
    """The id of the run the train step created under ``run_name``.

    Read out of the store rather than scraped off the step's stdout: the run id
    is state the store holds, and reading it there is one less thing coupled to
    how a step words its log.
    """
    with reading(store) as client:
        experiment = client.get_experiment_by_name(EXPERIMENT)
        assert experiment is not None, f"no experiment {EXPERIMENT!r} in {store.uri}"
        runs = client.search_runs(
            [experiment.experiment_id], filter_string=f"tags.`mlflow.runName` = '{run_name}'"
        )
    assert len(runs) == 1, f"expected one run named {run_name!r}, found {len(runs)}"
    return runs[0].info.run_id


def registered_versions(store: Store, model_name: str = MODEL_NAME) -> list[int]:
    """Every version registered under ``model_name``, smallest first.

    This is the measurement the whole module turns on. A refusal is not "the log
    said no"; it is this list being the same list afterwards.
    """
    with reading(store) as client:
        versions = client.search_model_versions(f"name = '{model_name}'")
    return sorted(int(version.version) for version in versions)


def train_and_promote(store: Store, run_name: str, data: Path = HEALTHY_DATA) -> str:
    """Drive all four steps for one candidate and return its run id.

    Used to establish the incumbent the gate later measures a candidate against.
    A registry with nothing in it promotes everything - "no incumbent" is the
    one case the gate cannot reject - so a rejection is only reachable once a
    version exists.
    """
    results = drive(
        store,
        [
            ("validate", lambda: ["--data", str(data)]),
            ("train", lambda: ["--data", str(data), "--run-name", run_name]),
            ("evaluate", lambda: ["--run-id", run_id_named(store, run_name)]),
            ("register", lambda: ["--run-id", run_id_named(store, run_name)]),
        ],
    )
    assert steps_that_ran(results) == ["validate", "train", "evaluate", "register"], report(results)
    assert not any(result.failed() for result in results), report(results)
    return run_id_named(store, run_name)


@pytest.fixture(scope="module")
def store(tmp_path_factory) -> Store:
    """A tracking store of this module's own, outside the checkout entirely."""
    return Store(tmp_path_factory.mktemp("refusal-paths-store"))


@pytest.fixture(scope="module", autouse=True)
def the_checkout_is_left_alone():
    """Fail the module if a step wrote a store into the repository.

    Only newly created files are an offence: a developer who has trained a model
    by hand has a legitimate ``mlruns/`` and ``mlflow.db`` in their checkout, and
    this must not turn that into a failure.
    """

    def footprint() -> dict[str, bool]:
        return {name: (REPO_ROOT / name).exists() for name in (DATABASE, ARTIFACTS)}

    before = footprint()
    yield
    created = [name for name, exists in footprint().items() if exists and not before[name]]
    assert not created, f"the pipeline wrote {created} into the checkout at {REPO_ROOT}"


class Sequence(NamedTuple):
    """A drive of the four steps, with the registry measured either side of it."""

    results: list[StepResult]
    versions_before: list[int]
    versions_after: list[int]
    run_id: str


@pytest.fixture(scope="module")
def incumbent(store: Store) -> str:
    """A registered version 1, so that the gate has something to compare against."""
    return train_and_promote(store, BASELINE)


@pytest.fixture(scope="module")
def rejection(store: Store, incumbent: str) -> Sequence:
    """The four steps driven for a candidate held to a margin it cannot clear.

    The whole sequence runs once, here, and the tests below read what it did:
    the registry is photographed immediately before and immediately after, so no
    test's assertion depends on which other test happened to run first.
    """
    before = registered_versions(store)
    results = drive(
        store,
        [
            ("validate", lambda: ["--data", str(HEALTHY_DATA)]),
            ("train", lambda: ["--data", str(HEALTHY_DATA), "--run-name", CANDIDATE]),
            (
                "evaluate",
                lambda: [
                    "--run-id",
                    run_id_named(store, CANDIDATE),
                    "--min-improvement",
                    UNCLEARABLE_MARGIN,
                ],
            ),
            ("register", lambda: ["--run-id", run_id_named(store, CANDIDATE)]),
        ],
    )
    return Sequence(
        results=results,
        versions_before=before,
        versions_after=registered_versions(store),
        run_id=run_id_named(store, CANDIDATE),
    )


@pytest.fixture(scope="module")
def unevaluated_run(store: Store, incumbent: str) -> str:
    """A trained candidate the gate has never seen, left deliberately untagged."""
    results = drive(
        store,
        [
            ("validate", lambda: ["--data", str(HEALTHY_DATA)]),
            ("train", lambda: ["--data", str(HEALTHY_DATA), "--run-name", UNEVALUATED]),
        ],
    )
    assert not any(result.failed() for result in results), report(results)
    return run_id_named(store, UNEVALUATED)


def test_the_four_steps_run_in_sequence_and_register_a_version(store: Store, incumbent: str):
    """The healthy path, without which the refusals below would prove nothing.

    A pipeline that refuses everything also never registers a rejected model.
    """
    assert registered_versions(store) == [1]
    assert store.database.is_file(), "the registry needs a database behind it"


def test_corrupted_data_fails_validation_and_training_never_runs(tmp_path):
    """The first refusal: a bad dataset costs nothing but the validate step.

    That training did not run is not taken on the word of the log. The steps are
    driven the way a pipeline drives them, and the proof is negative and total:
    the store this pipeline was pointed at is still an empty directory. A train
    step that ran would have created the database and the artifact directory -
    the corrupt dataset's defect is an impossible ``mpg``, which no estimator
    refuses to fit - so an empty directory is a training run that never started.
    """
    store = Store(tmp_path)

    results = drive(
        store,
        [
            ("validate", lambda: ["--data", str(CORRUPT_DATA)]),
            ("train", lambda: ["--data", str(CORRUPT_DATA), "--run-name", CANDIDATE]),
            ("evaluate", lambda: ["--run-id", run_id_named(store, CANDIDATE)]),
            ("register", lambda: ["--run-id", run_id_named(store, CANDIDATE)]),
        ],
    )

    assert steps_that_ran(results) == ["validate"], report(results)
    assert results[0].failed(), report(results)
    assert sorted(path.name for path in store.directory.iterdir()) == [], (
        "the contract stopped the pipeline, so nothing should have been recorded: "
        f"{sorted(path.name for path in store.directory.iterdir())}"
    )


def test_the_gate_rejects_a_candidate_that_cannot_clear_the_margin(rejection: Sequence):
    """The second refusal, mechanism one: a non-zero exit stops the pipeline."""
    evaluated = rejection.results[-1]

    assert steps_that_ran(rejection.results) == ["validate", "train", "evaluate"], report(
        rejection.results
    )
    assert evaluated.step == "evaluate"
    assert evaluated.failed(), report(rejection.results)


def test_a_rejected_candidate_leaves_the_registry_exactly_as_it_found_it(rejection: Sequence):
    """The property itself, measured where it matters rather than read in a log."""
    assert rejection.versions_after == rejection.versions_before
    assert rejection.versions_after == [1], "only the promoted baseline may be registered"


def test_registering_a_rejected_run_directly_is_refused(store: Store, rejection: Sequence):
    """Mechanism two: the verdict travels on the run, so refusal outlives the runner.

    This is the step run by hand, out of order, against a run the gate has
    already turned down - the case the pipeline's ``needs:`` chain cannot cover
    because there is no pipeline.
    """
    before = registered_versions(store)

    refused = run_step(store, "register", "--run-id", rejection.run_id)

    assert refused.failed(), refused.report()
    assert registered_versions(store) == before, "a rejected run reached the registry"


def test_registering_a_run_the_gate_never_saw_is_refused(store: Store, unevaluated_run: str):
    """No verdict is not a pass. Skipping the gate must not be a way through it."""
    before = registered_versions(store)

    refused = run_step(store, "register", "--run-id", unevaluated_run)

    assert refused.failed(), refused.report()
    assert registered_versions(store) == before, "an ungated run reached the registry"


def test_nothing_was_written_outside_the_temporary_store(store: Store):
    """Where the three training runs actually landed, asserted rather than assumed.

    The other half of this - that the checkout gained nothing - is asserted on
    teardown by :func:`the_checkout_is_left_alone`, because it can only be known
    once every test in the module has finished.
    """
    assert store.database.is_file(), "the runs, the gate's verdicts and the registry"
    assert (store.directory / ARTIFACTS).is_dir(), "the logged model artifacts"
    assert not store.directory.is_relative_to(
        REPO_ROOT
    ), f"a throwaway store must not be inside the checkout: {store.directory}"
