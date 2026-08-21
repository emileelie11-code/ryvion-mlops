"""The HTTP surface of the hand-built serving container.

Four endpoints, and none of them knows anything about automobiles:

``POST /predict``
    Takes records of named columns exactly as the data has them and returns one
    prediction per record.
``GET /healthz``
    Liveness. The process is up and answering. This is what a Kubernetes
    ``livenessProbe`` points at, and it deliberately does *not* depend on the
    model: a liveness probe that fails because a model is missing asks the
    orchestrator to restart a container that will fail again in exactly the same
    way, which is a crash loop rather than a diagnosis.
``GET /readyz``
    Readiness. The model is loaded and this replica can serve traffic. This is
    what a ``readinessProbe`` points at; it answers 503 until the model is
    there, so the orchestrator keeps traffic away rather than restarting.
``GET /schema``
    The input contract, read off the artifact. Nothing here is written down by
    hand - the model carries it.

The service holds one strong opinion, and it is about failure. The model
pipeline imputes missing values, so a request that omits a column would not
error: it would be silently filled in with a training-set mean and scored as
though it were complete. A wrong prediction returned with a 200 is worse than
any error, so input that does not match the model's declared contract is
refused with 422 and a message naming what was wrong.

Rejection happens in two places, and both of them belong to the model rather
than to this file. First, every record is checked against the column names the
signature declares. Then MLflow's own schema enforcement runs when the model is
called, and refuses types it cannot safely convert. Neither check is written out
here, which is why this service keeps working when the signature changes.
"""

import logging
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, status
from mlflow.exceptions import MlflowException
from pydantic import BaseModel, ConfigDict, Field

from serving.loader import ServedModel, load_model, resolve_model_uri

LOGGER = logging.getLogger("serving")

#: The loaded model, or ``None`` while it is missing or unloadable.
_model: ServedModel | None = None

#: Why it is missing, so that readiness can say so instead of just failing.
_load_failure: str | None = None


@asynccontextmanager
async def lifespan(_: FastAPI):
    """Load the model once, at startup, rather than on every request.

    A failure to load is logged and remembered rather than raised. The process
    stays up and answers ``/healthz``, ``/readyz`` answers 503 with the reason,
    and an operator gets a running container to ask questions of instead of a
    restart loop with the answer buried in a previous instance's logs.
    """
    global _model, _load_failure

    uri = resolve_model_uri()
    try:
        _model = load_model(uri)
        _load_failure = None
        LOGGER.info("loaded model from %s", uri)
    except Exception as failure:  # noqa: BLE001 - any load failure is the same failure here
        _model = None
        _load_failure = f"could not load a model from {uri}: {failure}"
        LOGGER.error(_load_failure)
    yield


app = FastAPI(
    title="Automobile MPG serving",
    summary="The hand-built serving container for the ryvion-mlops teaching repository.",
    description=(
        "Predicts miles per gallon from a raw record of named columns. The model is a "
        "complete scikit-learn pipeline loaded through its MLflow flavour, so it carries "
        "its own preprocessing and this service duplicates none of it. `GET /schema` "
        "reports the input contract the artifact declares."
    ),
    version="0.1.0",
    lifespan=lifespan,
)


class PredictionRequest(BaseModel):
    """One or more raw records, keyed by the column names the model declares."""

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "example": {
                "records": [
                    {
                        "cylinders": 8,
                        "displacement": 307.0,
                        "horsepower": "130.0",
                        "weight": 3504,
                        "acceleration": 12.0,
                        "model year": 70,
                        "origin": 1,
                        "car name": "chevrolet chevelle malibu",
                    }
                ]
            }
        },
    )

    records: list[dict[str, Any]] = Field(
        ...,
        min_length=1,
        description="Records exactly as the data has them. Call GET /schema for the columns.",
    )


class PredictionResponse(BaseModel):
    """One prediction per record, in the order the records were sent."""

    predictions: list[float]


def served_model() -> ServedModel:
    """The loaded model, or a 503 explaining why there is not one."""
    if _model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=_load_failure or "no model is loaded",
        )
    return _model


def reject_records_that_break_the_contract(
    model: ServedModel, records: list[dict[str, Any]]
) -> None:
    """Refuse anything the model's declared columns do not account for.

    Missing required columns would be imputed into a plausible-looking wrong
    answer; an unknown column is almost always a misspelt required one, and
    MLflow would drop it without comment. Both are named back to the caller.
    """
    if not model.declares_named_columns:
        return

    problems: list[str] = []
    for position, record in enumerate(records):
        missing = model.missing_columns(record)
        if missing:
            problems.append(f"record {position} is missing required {_columns(missing)}")
        unknown = model.unknown_columns(record)
        if unknown:
            problems.append(f"record {position} carries undeclared {_columns(unknown)}")

    if problems:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=(
                f"{'; '.join(problems)}. The model declares "
                f"{_columns(model.column_names)}; see GET /schema."
            ),
        )


def _columns(names: tuple[str, ...]) -> str:
    plural = "column" if len(names) == 1 else "columns"
    return f"{plural} {', '.join(repr(name) for name in names)}"


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest) -> PredictionResponse:
    """Score raw records against the model, or say clearly why they were refused."""
    model = served_model()
    reject_records_that_break_the_contract(model, request.records)

    try:
        predictions = model.predict(request.records)
    except (MlflowException, ValueError) as failure:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"the model refused this input: {failure}",
        ) from failure

    return PredictionResponse(predictions=predictions)


@app.get("/healthz")
def healthz() -> dict[str, str]:
    """Liveness: the process is up. Suitable as a Kubernetes ``livenessProbe`` target."""
    return {"status": "ok"}


@app.get("/readyz")
def readyz() -> dict[str, str]:
    """Readiness: a model is loaded. Suitable as a ``readinessProbe`` target."""
    model = served_model()
    return {"status": "ready", "model_uri": model.uri}


@app.get("/schema")
def schema() -> dict[str, Any]:
    """The input contract, read off the artifact rather than written down here."""
    return served_model().describe()
