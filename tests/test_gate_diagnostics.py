"""What the gate says when it finds no incumbent.

There are two ways to find nothing in a registry. One is a project's genuine
first model; the other is a store that is not the one the operator meant, most
often because the tracking destination is read per process and does not follow
anybody into a second terminal. Both promote, and both should - refusing the
first run would deadlock a new project - so the exit code cannot separate them
and the output has to.

These tests hold that separation in place. They drive the two entrypoints with
the tracking adapter stubbed out, so they assert on what a person reading the
log sees and nothing else. Delete the diagnostic and they fail.
"""

import pytest

from automobile import tracking
from automobile.entrypoints import evaluate, register
from automobile.tracking import RegistrySurvey

PROJECT_STORE = RegistrySurvey(
    uri="sqlite:///mlflow.db",
    from_environment=True,
    model_names=("automobile-mpg", "spare-parts-demand"),
)

THROWAWAY_STORE = RegistrySurvey(
    uri="file:///C:/Users/student/scratch/mlruns",
    from_environment=False,
    model_names=(),
)


class FakePage(list):
    """A page of results shaped like MLflow's, so paging can be exercised."""

    def __init__(self, items, token=None):
        super().__init__(items)
        self.token = token


class FakeRegisteredModel:
    def __init__(self, name):
        self.name = name


class FakeClient:
    """Stands in for MlflowClient: hands back pages, or raises what it was given."""

    def __init__(self, pages):
        self.pages = pages

    def search_registered_models(self, max_results=None, page_token=None):
        if isinstance(self.pages, Exception):
            raise self.pages
        if page_token is None:
            return self.pages[0]
        return self.pages[1]


def flat(text):
    """One line, so an assertion is not hostage to where the wrapper broke."""
    return " ".join(text.split())


def evaluate_with(monkeypatch, survey, *, incumbent=None, candidate=None):
    """Run the evaluate step against a stubbed registry and capture its exit code."""
    monkeypatch.setattr(evaluate, "survey_registry", lambda: survey)
    monkeypatch.setattr(evaluate, "run_metrics", lambda run_id: candidate or {"test_mse": 11.4})
    monkeypatch.setattr(evaluate, "find_incumbent", lambda model_name: incumbent)
    monkeypatch.setattr(
        evaluate,
        "record_gate_decision",
        lambda run_id, *, promote, reason, metric: "promote" if promote else "reject",
    )
    return evaluate.main(["--run-id", "run-under-test"])


def register_with(monkeypatch, survey, *, version="1", verdict=tracking.PROMOTED):
    """Run the register step against a stubbed registry and capture its exit code."""
    monkeypatch.setattr(register, "survey_registry", lambda: survey)
    monkeypatch.setattr(register, "read_gate_decision", lambda run_id: verdict)
    monkeypatch.setattr(register, "run_metrics", lambda run_id: {"test_mse": 11.4})
    monkeypatch.setattr(register, "run_params", lambda run_id: {"data_source": "data/auto.csv"})
    monkeypatch.setattr(
        register.mlflow,
        "register_model",
        lambda uri, name, tags=None: _version(name, version),
    )
    return register.main(["--run-id", "run-under-test"])


def _version(name, version):
    stub = FakeRegisteredModel(name)
    stub.version = version
    return stub


def test_no_incumbent_in_the_project_store_names_the_store_and_counts_what_is_in_it(
    monkeypatch, capsys
):
    exit_code = evaluate_with(monkeypatch, PROJECT_STORE)
    out = capsys.readouterr().out

    assert exit_code == 0
    assert PROJECT_STORE.uri in out, "the destination consulted must be named"
    assert "2 registered model(s)" in out, "the reader must be told how much is there"
    assert "spare-parts-demand" in out


def test_no_incumbent_in_an_empty_store_says_it_is_empty_and_why_that_may_be_wrong(
    monkeypatch, capsys
):
    exit_code = evaluate_with(monkeypatch, THROWAWAY_STORE)
    out = capsys.readouterr().out

    assert exit_code == 0, "a first-ever run still promotes"
    assert THROWAWAY_STORE.uri in out, "the destination consulted must be named"
    assert "0 registered models" in out
    assert tracking.ENV_TRACKING_URI in out, "name the setting that decides the destination"
    assert "per process" in flat(out), "the trap is that the setting does not follow you"


def test_the_two_ways_of_finding_nothing_do_not_read_alike(monkeypatch, capsys):
    """The point of the whole slice: one glance, no second command."""
    assert evaluate_with(monkeypatch, PROJECT_STORE) == 0
    populated = capsys.readouterr().out

    assert evaluate_with(monkeypatch, THROWAWAY_STORE) == 0
    empty = capsys.readouterr().out

    assert populated != empty
    assert PROJECT_STORE.uri in populated and PROJECT_STORE.uri not in empty
    assert THROWAWAY_STORE.uri in empty and THROWAWAY_STORE.uri not in populated
    assert "0 registered models" in empty and "0 registered models" not in populated


def test_a_promotion_with_nothing_to_compare_is_marked_louder_than_an_ordinary_one(
    monkeypatch, capsys
):
    assert evaluate_with(monkeypatch, THROWAWAY_STORE) == 0
    first_ever = capsys.readouterr().out

    incumbent = tracking.Incumbent(version=3, run_id="older-run", metrics={"test_mse": 12.0})
    assert evaluate_with(monkeypatch, PROJECT_STORE, incumbent=incumbent) == 0
    ordinary = capsys.readouterr().out

    assert evaluate.NOTHING_COMPARED in first_ever
    assert evaluate.NOTHING_COMPARED not in ordinary
    assert "version 3" in ordinary, "an ordinary promotion names what it beat"


def test_a_rejection_still_exits_one_and_still_names_the_store(monkeypatch, capsys):
    incumbent = tracking.Incumbent(version=3, run_id="older-run", metrics={"test_mse": 1.0})

    exit_code = evaluate_with(monkeypatch, PROJECT_STORE, incumbent=incumbent)
    out = capsys.readouterr().out

    assert exit_code == 1
    assert PROJECT_STORE.uri in out


def test_registering_the_first_version_says_what_the_store_held_before_it(monkeypatch, capsys):
    exit_code = register_with(monkeypatch, THROWAWAY_STORE, version="1")
    out = capsys.readouterr().out

    assert exit_code == 0
    assert register.FIRST_VERSION_HERE in out
    assert THROWAWAY_STORE.uri in out
    assert "0 registered models" in out
    assert tracking.ENV_TRACKING_URI in out
    assert "per process" in flat(out)


def test_registering_a_later_version_names_the_store_without_shouting(monkeypatch, capsys):
    exit_code = register_with(monkeypatch, PROJECT_STORE, version="4")
    out = capsys.readouterr().out

    assert exit_code == 0
    assert PROJECT_STORE.uri in out
    assert "2 registered model(s)" in out
    assert register.FIRST_VERSION_HERE not in out


def test_refusing_an_unevaluated_run_names_the_store_it_looked_in(monkeypatch, capsys):
    """A run never evaluated, and one evaluated elsewhere, are the same silence."""
    exit_code = register_with(monkeypatch, THROWAWAY_STORE, verdict=None)
    err = capsys.readouterr().err

    assert exit_code == 1
    assert THROWAWAY_STORE.uri in err


def test_the_survey_reports_the_destination_and_whether_it_was_chosen(monkeypatch):
    monkeypatch.setenv(tracking.ENV_TRACKING_URI, "sqlite:///mlflow.db")
    monkeypatch.setattr(tracking.mlflow, "get_tracking_uri", lambda: "sqlite:///mlflow.db")
    monkeypatch.setattr(
        tracking,
        "MlflowClient",
        lambda: FakeClient([FakePage([FakeRegisteredModel("automobile-mpg")])]),
    )

    survey = tracking.survey_registry()

    assert survey.uri == "sqlite:///mlflow.db"
    assert survey.from_environment is True
    assert survey.model_names == ("automobile-mpg",)


def test_an_unset_destination_is_reported_as_the_default_not_as_a_choice(monkeypatch):
    monkeypatch.delenv(tracking.ENV_TRACKING_URI, raising=False)
    monkeypatch.setattr(tracking.mlflow, "get_tracking_uri", lambda: "file:///tmp/mlruns")
    monkeypatch.setattr(tracking, "MlflowClient", lambda: FakeClient([FakePage([])]))

    survey = tracking.survey_registry()

    assert survey.from_environment is False
    assert survey.is_empty
    assert "unset" in survey.destination()


def test_the_survey_pages_through_every_registered_model(monkeypatch):
    monkeypatch.setattr(tracking.mlflow, "get_tracking_uri", lambda: "sqlite:///mlflow.db")
    pages = [
        FakePage([FakeRegisteredModel("one"), FakeRegisteredModel("two")], token="next"),
        FakePage([FakeRegisteredModel("three")]),
    ]
    monkeypatch.setattr(tracking, "MlflowClient", lambda: FakeClient(pages))

    assert tracking.survey_registry().model_names == ("one", "two", "three")


def test_a_store_that_will_not_list_its_models_is_reported_rather_than_raised(monkeypatch):
    monkeypatch.setattr(tracking.mlflow, "get_tracking_uri", lambda: "sqlite:///mlflow.db")
    monkeypatch.setattr(tracking, "MlflowClient", lambda: FakeClient(RuntimeError("no registry")))

    survey = tracking.survey_registry()

    assert survey.model_names is None
    assert "no registry" in survey.contents()
    assert "unverified" in survey.caution("automobile-mpg")


@pytest.mark.parametrize(
    "survey, expected",
    [
        (THROWAWAY_STORE, "per process"),
        (PROJECT_STORE._replace(model_names=("something-else",)), "--model-name"),
        (PROJECT_STORE, "no version of it"),
    ],
)
def test_the_caution_matches_what_was_actually_found(survey, expected):
    assert expected in flat(survey.caution("automobile-mpg"))
