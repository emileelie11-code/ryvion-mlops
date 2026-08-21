"""The serving model loader: how the container gets a model, and its contract.

There is one rule in this module and the whole slice depends on it: **no
preprocessing happens here**. The artifact training logs is a complete
scikit-learn pipeline - coercion, imputation, scaling and estimator - carrying
the statistics it learned at fit time. It is loaded through its MLflow flavour
and handed a raw row exactly as the data has it, sentinel ``?`` included.

That is what closes the training/serving skew this repository inherited. The
predecessor imputed missing horsepower values in a training script, threw the
statistics away, and left the caller of the scoring service to reproduce a
training-set mean it had never been told. If you ever find yourself adding a
``to_numeric``, a ``fillna`` or a column drop to this file, the model has stopped
carrying its own preprocessing and something upstream needs fixing instead.

The second thing this module does is read the model's **declared schema** and
keep it, so that the service can reject a record that does not match it. That
matters more than it looks: the pipeline imputes missing values, so a request
that quietly omits ``weight`` would not fail - it would be filled in with a
training-set mean and score as though nothing were wrong. A wrong prediction
returned with a 200 is the worst outcome available to this service, so a record
that does not carry every required column is refused before the model sees it.

Nothing here hard-codes the schema. The columns, their types and which of them
are required are read off the artifact at load time, so the service keeps
working when the model's signature changes - for instance when ``horsepower``
stops being typed as a string and becomes a nullable number.
"""

import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mlflow.pyfunc
import numpy as np
import pandas as pd

#: Overrides where the model is loaded from. Anything MLflow can resolve works:
#: a directory (what the image ships), or a ``models:/...`` URI when a tracking
#: store is configured.
ENV_MODEL_URI = "MODEL_URI"

#: The ``model/`` directory beside this module. The image copies the exported
#: artifact to exactly this place, so a container needs no configuration at all
#: to find its model, and a developer running uvicorn from a clone finds the
#: same one where the README told them to export it.
DEFAULT_MODEL_URI = str(Path(__file__).resolve().parent / "model")


def resolve_model_uri() -> str:
    """Where to load the model from: the environment's choice, else the default."""
    return os.environ.get(ENV_MODEL_URI) or DEFAULT_MODEL_URI


@dataclass(frozen=True)
class ServedModel:
    """A loaded model, together with the input contract it declares for itself."""

    #: Where it was loaded from, for the readiness endpoint to report.
    uri: str

    #: The MLflow ``pyfunc`` wrapper around the scikit-learn pipeline.
    pyfunc: mlflow.pyfunc.PyFuncModel

    #: Every column the signature declares, in signature order.
    column_names: tuple[str, ...]

    #: Those of them that every record must carry.
    required_columns: tuple[str, ...]

    @property
    def declares_named_columns(self) -> bool:
        """Whether this model's signature names its inputs.

        A model without a column schema cannot have its records checked against
        one; MLflow's own enforcement is then the only gate. Every model this
        repository trains declares one, so this is a guard rather than a path.
        """
        return bool(self.column_names)

    @property
    def optional_columns(self) -> tuple[str, ...]:
        """Declared columns a record may leave out - a nullable one, for instance."""
        return tuple(name for name in self.column_names if name not in self.required_columns)

    def missing_columns(self, record: Mapping[str, Any]) -> tuple[str, ...]:
        """Required columns this record does not carry."""
        return tuple(name for name in self.required_columns if name not in record)

    def unknown_columns(self, record: Mapping[str, Any]) -> tuple[str, ...]:
        """Columns this record carries that the model does not declare."""
        return tuple(name for name in record if name not in self.column_names)

    def to_frame(self, records: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
        """Turn the request's records into the frame the model is called with.

        Column *order* is restored from the signature - and that is the only
        thing done to the data. No coercion, no filling, no dropping.
        """
        frame = pd.DataFrame(list(records))
        if not self.declares_named_columns:
            return frame
        return frame[[name for name in self.column_names if name in frame.columns]]

    def predict(self, records: Sequence[Mapping[str, Any]]) -> list[float]:
        """Score the records, raising if the model's own schema enforcement refuses them."""
        predictions = self.pyfunc.predict(self.to_frame(records))
        return [float(value) for value in np.asarray(predictions).ravel()]

    def describe(self) -> dict[str, Any]:
        """The contract the artifact declares, as the ``/schema`` endpoint reports it.

        Serving this is a small thing that makes a large point: the caller does
        not have to be told the input schema out of band, because the model
        carries it.
        """
        schema = self.pyfunc.metadata.get_input_schema()
        return {
            "model_uri": self.uri,
            "columns": schema.to_dict() if schema is not None else None,
            "required_columns": list(self.required_columns),
            "optional_columns": list(self.optional_columns),
        }


def load_model(uri: str | None = None) -> ServedModel:
    """Load the model through its MLflow flavour and read the contract off it."""
    resolved = uri or resolve_model_uri()
    pyfunc = mlflow.pyfunc.load_model(resolved)
    if pyfunc is None:
        # MLflow wraps load_model in its tracing machinery, and when that
        # machinery cannot reach the tracking store it warns, swallows the
        # result and hands back None rather than raising. A None here would
        # otherwise surface later as a mystifying AttributeError, so it is
        # turned into the failure it actually is. The image sets
        # MLFLOW_TRACKING_URI at a writable path so this does not arise.
        raise RuntimeError(
            f"MLflow returned no model for {resolved!r}. This usually means it could not "
            "reach its tracking store - check MLFLOW_TRACKING_URI points somewhere "
            "writable, and read the MLflow warnings above."
        )

    schema = pyfunc.metadata.get_input_schema()
    if schema is None or not schema.has_input_names():
        return ServedModel(uri=resolved, pyfunc=pyfunc, column_names=(), required_columns=())

    column_names = tuple(schema.input_names())
    required = tuple(spec.name for spec in schema.inputs if getattr(spec, "required", True))
    return ServedModel(
        uri=resolved,
        pyfunc=pyfunc,
        column_names=column_names,
        required_columns=required,
    )
