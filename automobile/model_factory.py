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

from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from automobile.dataset import NUMERIC_FEATURES

#: The directory of the package the fitted model needs in order to unpickle.
PACKAGE_ROOT = Path(__file__).resolve().parent

#: The name that directory is imported - and, when it is installed at all,
#: distributed - under. See :func:`model_code_paths` for why the model artifact
#: must not name it as a dependency.
PACKAGE_NAME = PACKAGE_ROOT.name


def model_code_paths() -> list[str]:
    """The source that must be logged *inside* the model artifact.

    :func:`build_pipeline` puts :func:`coerce_to_numeric` into a
    ``FunctionTransformer``, and cloudpickle serialises a module-level function
    **by reference**: the artifact records the string
    ``automobile.model_factory.coerce_to_numeric`` and nothing else. Unpickling
    it therefore imports this package, and anywhere the source tree is absent
    that import fails with ``ModuleNotFoundError: No module named 'automobile'``.

    Passing this to ``log_model(..., code_paths=...)`` copies the package into
    the artifact's own ``code/`` directory, which MLflow prepends to
    ``sys.path`` before it unpickles. The dependency then travels *inside* the
    model instead of being an unstated requirement of whatever environment
    happens to load it - so the model loads in a no-code deployment that builds
    its own container from the registry, and the serving image no longer has to
    copy the package in as a workaround.

    The whole package is carried rather than this one module, because
    :mod:`automobile.dataset` is imported at module scope here and the import
    is spelled absolutely; a single-file code path would resolve the transformer
    and then fail one import deeper.

    Carrying the code is only half of it. MLflow infers the artifact's pip
    requirements by loading it and reading back what it imported, so when this
    package happens to be installed in the training environment it is inferred
    as a requirement too - and ``automobile==0.1.0`` is a name no index can
    resolve. That is the same defect wearing different clothes: a dependency
    stated as a name the loader is expected to find for itself. The train step
    strikes it back out, using :data:`PACKAGE_NAME`, once the code is inside the
    artifact and the requirement is therefore false as well as unmeetable.
    """
    return [str(PACKAGE_ROOT)]


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
