"""The model factory: one place where the model is constructed.

Training, any future retraining and the serving container all obtain the model
from :func:`build_pipeline`, so none of them can drift from the others. That is
the point of the extraction, and it is what closes the training/serving skew
this repository inherited: the predecessor imputed missing horsepower values in
a script, threw the statistics away, and left whoever called the scoring service
to reproduce a training-set mean it had never been told.

Here, imputation and scaling are *stages of the model*. They are fitted when the
model is fitted, they are serialised when the model is serialised, and the
caller supplies nothing but a raw row.

Two details are deliberate rather than incidental:

``StandardScaler``
    stands in for the ``normalize=True`` constructor argument the predecessor
    passed to ``LinearRegression``. scikit-learn removed that parameter in 1.2,
    which - with the dependency unpinned - is what took the training step down.
    The scaler is the supported way to say the same thing.

``remainder="drop"``
    is how the free-text ``car name`` column leaves the pipeline. It is dropped
    *inside* the model rather than before it, so that the model's declared input
    schema is the shape of the data as it really arrives.

The estimator is a plain linear regression, and that is not an oversight: it
fits in milliseconds, it is explainable in one sentence, and this is a course
about operations rather than modelling.
"""

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from automobile.dataset import NUMERIC_FEATURES


def coerce_to_numeric(frame: pd.DataFrame) -> pd.DataFrame:
    """Turn every column into a number, sentinel values becoming missing values.

    A module-level function rather than a lambda on purpose: the fitted pipeline
    is serialised into the model artifact, and a lambda cannot be pickled.
    """
    return pd.DataFrame(frame).apply(pd.to_numeric, errors="coerce")


def build_pipeline() -> Pipeline:
    """Return a new, unfitted model pipeline.

    Three stages, in the order a raw row passes through them:

    ``prep``
        coerce the numeric columns (``?`` becomes missing), fill what is missing
        with the column mean learned at fit time, and drop everything else -
        which is to say, the free-text ``car name``.
    ``scale``
        standardise the features.
    ``model``
        the estimator.
    """
    numeric = Pipeline(
        steps=[
            ("coerce", FunctionTransformer(coerce_to_numeric, feature_names_out="one-to-one")),
            ("impute", SimpleImputer(strategy="mean")),
        ]
    )
    preprocessing = ColumnTransformer(
        transformers=[("numeric", numeric, list(NUMERIC_FEATURES))],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    return Pipeline(
        steps=[
            ("prep", preprocessing),
            ("scale", StandardScaler()),
            ("model", LinearRegression()),
        ]
    )
