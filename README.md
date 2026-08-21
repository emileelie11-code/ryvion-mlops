# ryvion-mlops

The worked example for a 35-hour graduate MLOps module: an automobile price
regression, taken from raw data to a served prediction, with the operational
machinery around it visible rather than hidden. Students fork this repository
and submit their coursework as pull requests against their own fork.

It is a rebuild. Its predecessor was a five-year-unmaintained fork of Microsoft's
`MLOpsPython` reference architecture, and it no longer ran. One of the reasons it
stopped running is the reason for the strictest rule here: **a single unpinned
dependency**. Nobody changed the training code; scikit-learn 1.2 removed a
constructor parameter it passed, and the training step died. Every dependency in
this repository is pinned to an exact version, and a unit test fails the build if
one is not.

---

## Requirements

- **Python 3.11.** Not 3.12 or newer - the package declares `requires-python`
  as `==3.11.*` and `pip install` will refuse anything else. 3.11 is the widest
  overlap between Azure ML's curated images, `azure-ai-ml` and MLflow.
- `git`.

Nothing else to train the model: it runs on a laptop with no cloud account and
no credentials. A **container engine** is needed for one section only - building
and running the serving image - and that section still needs no cloud account.

## Quickstart - from a fresh clone to a green test run

```bash
git clone https://github.com/emileastih1/ryvion-mlops.git
cd ryvion-mlops

# 1. A virtual environment on Python 3.11.
python3.11 -m venv .venv          # Windows: py -3.11 -m venv .venv
source .venv/bin/activate         # Windows: .venv\Scripts\activate

# 2. The locked environment - the exact versions CI validated.
python -m pip install --upgrade pip
python -m pip install -r environments/dev.lock.txt

# 3. This repository's own package, without touching those versions.
python -m pip install -e . --no-deps

# 4. The same two commands both CI systems run.
python -m flake8 .
python -m pytest
```

The last command should end in `150 passed`. If it does, your environment matches
the one the pull-request gate uses.

Then train the model, which needs no cloud account and no credentials:

```bash
python -m automobile.entrypoints.train
```

It reads `data/auto-mpg.csv`, fits the pipeline, and finishes in a few seconds
by printing the run it created:

```
tracking uri:    file:///.../ryvion-mlops/mlruns
experiment:      automobile-mpg
run id:          6855a100cc4644dfbfd9d3422f79624f
model uri:       models:/m-fbaecf2e2b3344e2a9d2202d97446a1d
```

`mlflow ui` in the same directory will show the run, its parameters, its
metrics and the logged model.

## What runs today

The four pipeline steps exist as command-line entrypoints, installed by step 3
above:

| Command | Module | Step |
|---|---|---|
| `automobile-validate` | `automobile.entrypoints.validate` | Check the data against its contract |
| `automobile-train` | `automobile.entrypoints.train` | Fit the model pipeline and log the artifact |
| `automobile-evaluate` | `automobile.entrypoints.evaluate` | Apply the promote-or-reject gate |
| `automobile-register` | `automobile.entrypoints.register` | Register the promoted model |

All four are implemented. `--help` works on all four, and the test suite holds
that surface in place.

Run end to end, the pipeline looks like this. Note the tracking destination: the
registry that `evaluate` and `register` use needs a database behind it, and
locally that is one SQLite file and no account.

```bash
export MLFLOW_TRACKING_URI=sqlite:///mlflow.db   # Windows: set MLFLOW_TRACKING_URI=sqlite:///mlflow.db

python -m automobile.entrypoints.validate        # step zero: the data contract
python -m automobile.entrypoints.train           # prints the run id
python -m automobile.entrypoints.evaluate --run-id RUN_ID
python -m automobile.entrypoints.register --run-id RUN_ID
```

Both gates report themselves the same way: `validate` exits non-zero when the
data breaks its contract, and `evaluate` exits **0** when it promotes the
candidate and **1** when it rejects it, so a run stops before it reaches
`register`.

## The data contract

`automobile/data_contract.py` holds every rule about this dataset behind one
call, `validate(frame)`, which returns a report rather than raising. The
entrypoint turns that report into an exit code, so a bad dataset costs one
command instead of a training run:

```bash
python -m automobile.entrypoints.validate                 # the seed dataset: exits 0
python -m automobile.entrypoints.validate --data broken.csv
```

```
FAIL  automobile-mpg data contract: 1 of 8 rules violated over 398 rows.

mpg_is_positive [mpg]
    every 'mpg' must be greater than zero
    2 offending row(s): 3, 41
    offending value(s): '-1.0'
```

Every violation is reported in one pass, and a row-level rule names the rows.
The rules are written with [Pandera](https://pandera.readthedocs.io/) against
defects that are genuinely in this data - the row count, the positive target, the
type of every column, and the six holes in `horsepower` - never against invented
ones.

### The `?` sentinel, and where it is dealt with

`?` is a 1983 fixed-width-file encoding artifact, not a domain concept.
`automobile.dataset.parse_sentinels` reads it as a missing value at the data
boundary, and `load_dataset` applies it, so `horsepower` reaches the model as a
**nullable `double`** rather than a string. The consequence is the served
contract: a caller sends `130.0`, or JSON `null` for "unknown", instead of
`"130.0"` and no way at all to say "unknown".

What does *not* move out to the boundary is the mean that fills those holes.
That statistic is fitted from the training data and stays inside the model
pipeline, where it is serialised into the artifact - moving it would reintroduce
exactly the training/serving skew this repository was rebuilt to close. The rule
is the line between the two: **stateless format parsing at the boundary, fitted
transforms inside the model.**

## The dataset

`data/auto-mpg.csv` is the canonical UCI *Auto MPG* dataset, committed as a seed
fixture: 398 cars, nine columns, and **six rows whose `horsepower` is `?`
instead of a number**. Those six are not dirt left behind by accident. They are
the dataset's real defect, they are what the data contract is written against,
and they are what the model pipeline survives without help from its caller. Do
not "clean" them out of the file - the loader reads `?` as a missing value on the
way in, which is a different thing from the file not having it.

The file is committed rather than downloaded because a classroom that depends on
an upstream host being reachable at 09:00 is a classroom that occasionally does
not happen. Later slices register it as a versioned data asset; the committed
file is the seed, and the asset version is what appears in a model's lineage.

Source: <https://archive.ics.uci.edu/dataset/9/auto+mpg>, file `auto-mpg.data`
(MD5 `b858f4580d0066c48e260dd3b96f1ed8`), converted to CSV with the header row
the notebook uses. No value was changed.

## The model artifact

The thing training logs is a **`sklearn.pipeline.Pipeline`**, not a bare
estimator:

| Stage | Does |
|---|---|
| `prep` | Coerces the numeric columns (`?` becomes missing), imputes with the column mean learned at fit time, and drops the free-text `car name` |
| `scale` | `StandardScaler` - the supported replacement for the `normalize=True` argument scikit-learn removed in 1.2, which is the argument that killed the predecessor |
| `model` | `LinearRegression` |

Because preprocessing is *inside* the model, its statistics are fitted and
serialised with it, and whoever calls the model supplies nothing but a raw row.
That is the training/serving skew the predecessor carried, closed. The model is
logged with an MLflow signature and an input example, so the artifact declares a
schema of named columns - and the input example deliberately includes a row whose
`horsepower` is missing, which is what makes the column `double (optional)` in
the signature and JSON `null` an acceptable value for it.

## The quality gate

Whether a candidate replaces the incumbent is a **pure function** in
`automobile/quality_gate.py`:

```python
decide(candidate_metrics, incumbent_metrics, policy) -> GateDecision
```

It reads no files, imports no tracking library, and imports nothing outside the
standard library at all - a unit test fails the build if that ever stops being
true. In the predecessor this decision was entangled with the platform: the
evaluation script compared two numbers and then reached up to cancel its own
parent pipeline run, which meant the policy could not be read, tested or changed
without a workspace.

Three behaviours are worth knowing before you change the threshold:

- **No incumbent promotes.** The first model ever trained has nothing to beat.
  A gate that refused it would deadlock the pipeline on its first run.
- **The boundary is inclusive.** A candidate landing exactly on the required
  value is promoted, so `--min-improvement` reads as "at least this much
  better".
- **The policy is configuration.** The default is mean squared error on the
  held-out half, minimised, with no margin demanded - and it is a default, not a
  rule:

  ```bash
  python -m automobile.entrypoints.evaluate --run-id RUN_ID       --metric test_r2 --goal maximise --min-improvement 0.01
  ```

The verdict leaves the step two ways, on purpose. It becomes the **exit code**,
which is what stops a pipeline - `0` promoted, `1` rejected. It is *also*
written onto the candidate run as the tag `automobile.gate.decision`, and
`register` refuses to register anything not tagged `promote`, including a run
the gate has never seen. An exit code is only as good as the thing reading it,
and `register` is also a command a person can type.

There is no override flag. A gate with a bypass is not a gate.

This replaces the predecessor's parent-run cancellation, and the semantics
differ deliberately: **a rejected run now reports as failed, not cancelled.**
Azure ML v2 has no equivalent of that cancellation, and an exit code is the
better lesson anyway.

## Where a run is recorded

The training code is backend-agnostic by construction: it reads its destination
from the environment and names no provider.

| Environment variable | Unset | Set |
|---|---|---|
| `MLFLOW_TRACKING_URI` | A local file store under `./mlruns`. No account, no credentials, no network. | Whatever it points at. |
| `MLFLOW_EXPERIMENT_NAME` | `automobile-mpg` | The experiment runs are grouped under. |

Pointing the identical code at a managed backend is therefore a change of
setting, not a change of code. There is no `if azure:` branch anywhere, and a
unit test fails the build if any module in the domain package imports a cloud
SDK.

### The registry needs a database

Recording runs is all `train` needs, and a directory is enough for that. The
**model registry** that `evaluate` reads the incumbent from and `register`
writes a version to wants somewhere it can hold a version counter, so point the
same variable at a database instead:

```bash
export MLFLOW_TRACKING_URI=sqlite:///mlflow.db
```

That is one file in your working directory, created on first use, ignored by
git, and still no account and no network. It is the **only** difference between
running the gate locally and running it against the managed workspace, where the
same variable points at the workspace instead. The destination stays
configuration - nothing in this repository branches on it, and nothing
hard-codes it.

`mlflow ui --backend-store-uri sqlite:///mlflow.db` shows the runs, the gate
tags and the registered versions together. To start over, delete `mlflow.db` and
`mlartifacts/`.

## The serving container

There are two deployment paths in this repository and the contrast between them
is the lesson. This is the first one: an image you build yourself, from a
Dockerfile short enough to read in one sitting. Later, the platform builds an
equivalent image from the model registry and you get to judge the trade. This
half needs a container engine and nothing else - no cloud account, no
credentials, no registry login - so the whole container-and-Kubernetes portion
of the course runs on a laptop.

### Build it and run it

```bash
# 1. Train. Note the model uri it prints on the last-but-two line.
python -m automobile.entrypoints.train

# 2. Export that model next to the Dockerfile. It is a build input rather than
#    something committed: this repository versions the code and the seed data,
#    and serving/model/ is gitignored.
mlflow artifacts download --artifact-uri models:/<the model id> --dst-path serving/model

# 3. Build, from the repository root - not from serving/. The Dockerfile needs
#    the requirements manifest and the domain package, which live above it.
docker build -f serving/Dockerfile -t ryvion-mlops-serving:local .

# 4. Run it.
docker run --rm -p 8000:8000 ryvion-mlops-serving:local
```

To run the same service without a container - which is how you debug it - export
the model as above and start it from the repository root:

```bash
python -m pip install -r environments/serving.requirements.txt
uvicorn serving.app:app --port 8000
```

Either way, `MODEL_URI` overrides where the model is loaded from. Unset, it is
the `model/` directory beside `serving/app.py`, which is exactly where the image
puts it and where step 2 above puts it in a clone.

### The endpoints

| Endpoint | Answers |
|---|---|
| `POST /predict` | One prediction per record. Records carry the column names the data has, raw. |
| `GET /healthz` | Liveness - the process is up. Point a Kubernetes `livenessProbe` here. |
| `GET /readyz` | Readiness - a model is loaded. Point a `readinessProbe` here; 503 until it is. |
| `GET /schema` | The input contract, read off the artifact rather than written down anywhere. |
| `GET /docs` | The generated API page, if you would rather click than curl. |

The two health endpoints are separate on purpose. A liveness probe that fails
because a model is missing asks the orchestrator to restart a container that
will fail in exactly the same way one second later; a readiness probe that fails
asks it to keep traffic away, which is the correct response and the one that
leaves a running container to ask questions of.

```bash
curl -s localhost:8000/healthz
# {"status":"ok"}

curl -s -X POST localhost:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{"records":[{"cylinders":8,"displacement":307.0,"horsepower":"130.0",
       "weight":3504,"acceleration":12.0,"model year":70,"origin":1,
       "car name":"chevrolet chevelle malibu"}]}'
# {"predictions":[14.975242297915628]}
```

Send `"horsepower": "?"` and you still get a prediction. That is the point of
the whole design: the sentinel is handled by the imputer *inside* the model,
with the mean it learned at training time, and this service does not know that
imputation is a thing that exists.

### Wrong input fails; it does not guess

```bash
# weight left out
{"detail":"record 0 is missing required column 'weight'. The model declares
 columns 'cylinders', 'displacement', ...; see GET /schema."}    # HTTP 422
```

This matters more than it looks. The model imputes missing values, so a request
that quietly omits `weight` would *not* error - it would be filled in with a
training-set mean and scored as though it were complete. A wrong prediction
returned with a 200 is the worst thing this service could do, so a record that
does not match the model's declared contract is refused before the model sees
it, and the reply names the column.

Two gates do that work, and neither is written out by hand. First the service
checks each record against the column names in the artifact's signature. Then
MLflow's own schema enforcement refuses types it cannot safely convert. Because
both are read from the model, they keep working when the signature changes.

### What is in the image, and what is not

- **Multi-stage.** The virtual environment is built in the first stage and
  copied into the second; the pip cache and the wheel downloads stay behind.
- **Non-root**, as uid/gid `10001`, written numerically because Kubernetes
  cannot resolve a user *name* against an image and a manifest that sets
  `runAsNonRoot` would refuse to start otherwise.
- **Pinned**, to `environments/serving.requirements.txt` and to a patch release
  of the base image. `--only-binary=:all:` fails the build rather than compiling
  anything from source, which is what keeps the image buildable on both x86 and
  Apple Silicon without a toolchain inside it.
- **No preprocessing.** The service loads a complete scikit-learn pipeline
  through its MLflow flavour and hands it a raw row. If you ever find a
  `to_numeric` or a `fillna` in `serving/`, the skew this repository exists to
  close has crept back in.
- **The `automobile` package**, which is there for one specific reason: the
  pipeline's first stage wraps a function from `automobile.model_factory`, and
  cloudpickle stores that function as a *reference*. Without the module on the
  path the artifact does not unpickle at all. The serving dependency manifest
  still carries no training, testing or cloud package.
- **`.dockerignore`** keeps the rest of the repository - your `.venv`, your
  `mlruns/`, the tests, the notebook - out of the build context entirely.

## Repository layout

```
automobile/            The domain package.
  entrypoints/         One argparse shell per pipeline step. No domain logic.
serving/               The hand-built serving application and its Dockerfile.
data/                  The seed dataset, committed. 398 rows, six of them defective.
environments/          The three dependency manifests, and the lockfile.
tests/                 Unit tests. No credentials, no network, no containers.
.github/workflows/     GitHub Actions: the always-on quality gate.
.pipelines/            Azure Pipelines: the definitions the course studies.
charts/                Helm charts for the A/B deployment material.
notebooks/             The exploratory notebook the course opens with.
docs/                  The PRD, and the conventions the agents in this repo follow.
```

## Dependencies

Three manifests, each with exactly one job. There is deliberately no fourth, and
no dependency list in `pyproject.toml` that would quietly become one.

| Manifest | Job |
|---|---|
| `environments/training.conda.yaml` | The Azure ML environment the four pipeline steps run in on cloud compute. Pins the interpreter as well as the libraries. |
| `environments/serving.requirements.txt` | The runtime of the hand-built serving container: load the model, answer prediction and health requests. |
| `environments/dev.requirements.txt` | Lint and test this repository, on a laptop and on a pull-request runner. |

`environments/dev.lock.txt` is the resolved output of the third, and it is what
you and CI actually install. It pins all 153 direct and transitive packages,
resolved with plain `pip` on Python 3.11 - no `uv`, no Poetry. Both are better
tools; both are one more thing to explain in a course that already carries a
container engine, an orchestrator, a package manager for it, a cloud ML
platform, a tracking library and a CI system.

The scientific libraries are pinned to the **same versions in all three**, so
that what CI tests is what training runs and what the serving container serves.
`tests/test_dependency_pins.py` enforces all of this: exact pins everywhere,
alignment across manifests, and exactly three manifests.

To change a dependency: edit `environments/dev.requirements.txt`, regenerate the
lockfile with the command in its header, and commit both.

## Continuous integration - two systems, on purpose

- **`.github/workflows/ci.yml`** is the always-on gate. It runs lint and unit
  tests on every pull request and needs no setup at all: fork the repository,
  open a pull request, and it runs. This is what gives coursework instant
  feedback.
- **`.pipelines/pr.yml`** is the same gate on the CI platform the module
  teaches, connected to a student's own organisation and their own fork. It is
  the artifact to read and extend, and it makes the split between code host and
  CI platform - two definitions, two systems, one repository - concrete.

Both run `flake8 .` and `pytest`. In the predecessor, the template holding those
two commands was commented out of the pipeline, so neither ever ran. Here it is
included.

## Carried over from the predecessor, unchanged

The Helm charts (`charts/abtest-model`, `charts/abtest-istio`), the load
generator (`charts/load_test.sh`), the Helm install and upgrade templates
(`.pipelines/helm-*.yml`) and the exploratory notebook (`notebooks/`) are copied
across byte-for-byte. Everything else from that repository was deleted by
design: the orchestration package, the duplicated scoring scripts, the R and
Databricks training path, and the parallel batch-scoring path.

## Where this is going

`docs/PRD-workstream-0-sdk-v2-rebuild.md` is the plan in full, and the open
issues are its slices: local training to a signed model artifact, the data
contract, the quality gate, the serving container, then the cloud pipeline,
managed endpoint and CI/CD definitions.
