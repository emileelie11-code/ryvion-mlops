# The hand-built serving image.
#
# This is the half of the course that says "I control the image". A later slice
# lets the platform build an equivalent one from the model registry, and the
# comparison between the two is a teaching objective rather than an accident -
# so every line here is meant to be readable, and none of it is magic.
#
# Build from the repository root, with a model exported into serving/model:
#
#     python -m automobile.entrypoints.train
#     mlflow artifacts download --artifact-uri models:/<id> --dst-path serving/model
#     docker build -f serving/Dockerfile -t ryvion-mlops-serving:local .
#     docker run --rm -p 8000:8000 ryvion-mlops-serving:local
#
# No cloud account, no credentials, no registry login. See the README for the
# whole walkthrough.

# Pinned to a patch release for the same reason every dependency in this
# repository is pinned: its predecessor died of a version that moved underneath
# it. Debian slim publishes native linux/amd64 and linux/arm64 builds, so an
# Apple Silicon laptop and an x86 CI runner both build this without emulation.
ARG PYTHON_IMAGE=python:3.11.16-slim-bookworm


# --- Stage 1: build the runtime environment -----------------------------------
#
# Everything expensive happens here - and nothing from this stage reaches the
# final image except the virtual environment itself. The pip cache, the wheel
# downloads and the requirements file all stay behind.
FROM ${PYTHON_IMAGE} AS builder

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

COPY environments/serving.requirements.txt /tmp/serving.requirements.txt

# --only-binary=:all: is a guard rather than an optimisation: it fails the build
# if any dependency would have to be compiled from source, which is what keeps
# this image buildable on both architectures without a toolchain in it.
RUN python -m venv /opt/venv \
    && /opt/venv/bin/pip install --upgrade pip \
    && /opt/venv/bin/pip install --only-binary=:all: -r /tmp/serving.requirements.txt


# --- Stage 2: the runtime -----------------------------------------------------
FROM ${PYTHON_IMAGE} AS runtime

LABEL org.opencontainers.image.title="ryvion-mlops serving" \
      org.opencontainers.image.description="Automobile MPG model served over HTTP." \
      org.opencontainers.image.source="https://github.com/emileastih1/ryvion-mlops"

# A dedicated account, referred to by number. Kubernetes cannot resolve a user
# *name* against an image, so a manifest that sets runAsNonRoot refuses to start
# a container whose USER is not numeric.
ARG UID=10001
ARG GID=10001
RUN groupadd --system --gid ${GID} serving \
    && useradd --system --uid ${UID} --gid ${GID} --home-dir /app --shell /usr/sbin/nologin serving

# MLFLOW_TRACKING_URI is set for one reason, and it is not tracking: this
# service records nothing. MLflow resolves its default file store relative to
# the working directory, and /app is root-owned and read-only to the account
# below - so the attempt to create /app/mlruns fails, and load_model then
# returns None instead of raising. Pointing it at a writable path avoids that.
# Under a read-only root filesystem, mount an emptyDir at /tmp.
ENV PATH=/opt/venv/bin:$PATH \
    PYTHONPATH=/app \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MLFLOW_TRACKING_URI=file:///tmp/mlruns \
    HOME=/app

COPY --from=builder /opt/venv /opt/venv

WORKDIR /app

# The application. Owned by root and read-only to the account that runs it:
# nothing in this service ever writes to disk.
COPY serving/__init__.py serving/loader.py serving/app.py /app/serving/

# The domain package, because the model artifact refers to it by name. The
# pipeline's first stage is a FunctionTransformer wrapping the model factory's
# numeric coercion, and cloudpickle serialises that function as a *reference* -
# so unpickling the model raises ModuleNotFoundError unless the module is
# importable. This is code the model itself depends on rather than training
# tooling: nothing here imports an entrypoint, and the serving requirements
# manifest still carries no training, testing or cloud dependency.
COPY automobile/ /app/automobile/

# The exported MLflow model directory. It is a build input, not something
# committed to the repository, so a build without it fails here - deliberately,
# and with the path it was looking for. An image whose model arrives at run time
# is an image Kubernetes cannot deploy on its own.
ARG MODEL_DIR=serving/model
COPY ${MODEL_DIR}/ /app/serving/model/

USER ${UID}:${GID}

EXPOSE 8000

# No curl, no wget: the runtime already has a Python that can make one request,
# and adding a tool to the image only to check on it is a poor trade. This is
# the container engine's own check; a Kubernetes probe points at the same path.
HEALTHCHECK --interval=30s --timeout=3s --start-period=30s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/healthz', timeout=2)"]

# Exec form, so uvicorn is PID 1 and receives the stop signal itself rather than
# through a shell that would swallow it.
CMD ["uvicorn", "serving.app:app", "--host", "0.0.0.0", "--port", "8000"]
