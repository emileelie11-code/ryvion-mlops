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
no credentials.

Two later sections need more, and neither of them needs a cloud account either:

- A **container engine** for building and running the serving image.
- **`kind` and `kubectl`** for deploying it to Kubernetes. kind runs a whole
  cluster as Docker containers, so the orchestration half of the course is also
  a laptop exercise. See [the same container, on
  Kubernetes](#the-same-container-on-kubernetes).

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

The last command should end in `173 passed, 7 deselected`. If it does, your
environment matches the one the pull-request gate uses.

The seven deselected are the integration test, which drives all four pipeline
steps as real subprocesses and takes tens of seconds rather than milliseconds.
It is kept out of the default run so that the run you type before every commit
stays fast, and CI runs it as a step of its own. To run it here:

```bash
python -m pytest -m integration
```

It needs no credentials and no network either - see [Proving the refusals, on
your own machine](#proving-the-refusals-on-your-own-machine).

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

That `export` is **per process.** It does not follow you into a second terminal,
a scheduled task or an IDE run configuration, and a shell without it does not
fail: MLflow falls back to a local file store, which has a registry of its own
and is empty. So `evaluate` and `register` say out loud which destination they
consulted and how many registered models were in it - see [The quality
gate](#the-quality-gate) - because a promotion against an empty store is a
promotion against nothing at all.

Both gates report themselves the same way: `validate` exits non-zero when the
data breaks its contract, and `evaluate` exits **0** when it promotes the
candidate and **1** when it rejects it, so a run stops before it reaches
`register`.

Those exit codes are the whole of the pipeline's control flow. `.github/workflows/pipeline.yml`
chains the same four commands as four dependent jobs - see [The pipeline, as a
CI workflow](#the-pipeline-as-a-ci-workflow) - and nothing in the entrypoints
knows whether it is being run by CI or by you.

## The data contract

`automobile/data_contract.py` holds every rule about this dataset behind one
call, `validate(frame)`, which returns a report rather than raising. The
entrypoint turns that report into an exit code, so a bad dataset costs one
command instead of a training run:

```bash
python -m automobile.entrypoints.validate                 # the seed dataset: exits 0
python -m automobile.entrypoints.validate --data data/auto-mpg-corrupt.csv
```

```
FAIL  automobile-mpg data contract: 2 of 8 rules violated over 398 rows.

horsepower_missing_count_is_the_documented_six [horsepower]
    exactly 6 rows may have a missing 'horsepower' - the documented defect, neither grown nor cleaned away

mpg_is_positive [mpg]
    every 'mpg' must be greater than zero
    1 offending row(s): 0
    offending value(s): '-18.0'
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

### `data/auto-mpg-corrupt.csv`

The same file with exactly two cells changed, committed so that the data
contract can be watched refusing something rather than only read about. Row
numbers are the ones the report prints:

| Row | Change | Rule it breaks |
|---|---|---|
| 0 | `mpg` `18.0` → `-18.0` | `mpg_is_positive` |
| 3 | `horsepower` `150.0` → `?` | `horsepower_missing_count_is_the_documented_six` |

Two defects, two rules, one pass - the second one being the interesting kind:
a *seventh* `?` is not obviously wrong to a human reading the file, and it is
caught only because the contract knows the documented number of holes is six.

Nothing trains on it. It exists for `--data data/auto-mpg-corrupt.csv`, on a
laptop or in the pipeline workflow, and it is the shortest honest answer to "how
do I know the gate works?"

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
  A gate that refused it would deadlock the pipeline on its first run. It is
  also the one verdict worth reading twice, so the step tells you where it
  looked - see below.
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

### "No incumbent" is two different things

Finding nothing in the registry means either *this project has never trained a
model* or *this is not the registry you meant*. Both promote, and both should -
but only one of them is a comparison you can trust, and the verdict is identical
in each. So `evaluate` and `register` name the destination they consulted and
count what is registered there:

```
tracking store:  sqlite:///mlflow.db  (MLFLOW_TRACKING_URI is set)
store contains:  1 registered model(s): automobile-mpg
incumbent:       none registered under 'automobile-mpg-v2'
decision:        promote (no-incumbent)
```

```
tracking store:  file:///C:/Users/you/scratch/mlruns  (MLFLOW_TRACKING_URI is unset, so this is MLflow's default store)
store contains:  0 registered models - this store has never held one
incumbent:       none registered under 'automobile-mpg'
decision:        promote (no-incumbent)
```

The second is the one that bites, and it costs nothing to fall into:
`MLFLOW_TRACKING_URI` is read **per process**, so a second terminal never
inherits the `export` you made in the first, quietly reads an empty file store,
and gets a green promotion for a model nothing was compared against. Both cases
also print a loud `*** NOTHING WAS COMPARED ***` banner, because a first-ever
promotion is not an ordinary one. `register` does the same on its side: it
reports the store and what it held, and marks a version `1` as the first thing
ever registered there.

None of this branches on which backend the destination happens to be. It reports
the URI MLflow resolved, whatever that turns out to be.

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

Note both halves of that: it is one file **in your working directory**, and the
variable is read **per process**. A different directory, or a shell that never
ran the export, is a different registry - an empty one, created on the spot,
that answers "nothing registered" perfectly truthfully. That is why the gate
prints the destination it used; see ["No incumbent" is two different
things](#no-incumbent-is-two-different-things).

The file is created on first use, ignored by git, and still no account and no
network. It is the **only** difference between running the gate locally and
running it against the managed workspace, where the same variable points at the
workspace instead. The destination stays
configuration - nothing in this repository branches on it, and nothing
hard-codes it.

`mlflow ui --backend-store-uri sqlite:///mlflow.db` shows the runs, the gate
tags and the registered versions together. To start over, delete `mlflow.db` and
`mlruns/`.

## The pipeline, as a CI workflow

`.github/workflows/pipeline.yml` runs the same four commands, in the same order,
on a machine that is not yours. It is triggered on every pull request, on every
push to `main`, and by hand from the **Run workflow** button.

```
1. Validate the data  ->  2. Train the candidate  ->  3. Apply the quality gate  ->  4. Register
```

Four **jobs**, not four steps in one job - which means four runners and four
empty filesystems. That is the expensive shape, and it was chosen on purpose:
one job with four `run:` lines would share a disk and make the state problem
disappear, and a student who only ever sees that learns that a pipeline is a
shell script. A pipeline stage is a machine, and machines do not share disks.

### How state moves between the stages

The steps are not independent. `train` produces a run, `evaluate` reads that
run's metrics and writes its verdict back onto it, and `register` reads both the
verdict and the model. All of that lives in the tracking store - which, because
the registry needs a database, is two things on disk:

| | |
|---|---|
| `mlflow.db` | the runs, their metrics, the gate's tags, the registry |
| `mlruns/` | the logged model artifacts themselves |

So two things travel between the jobs, by different means, because they are
different sizes:

| What | Size | How |
|---|---|---|
| The run id | one string | a **job output** - `outputs:` on `train`, `needs.train.outputs.run_id` on the jobs after it |
| The tracking store | a few megabytes | a **workflow artifact** - uploaded at the end of one job, downloaded at the start of the next |

Each job uploads the store under its own name rather than overwriting one, so
the run page carries three snapshots - `tracking-store-after-train`,
`-after-evaluate`, `-after-register`. Download the last one, point
`mlflow ui --backend-store-uri sqlite:///mlflow.db` at it, and you are looking at
what CI produced.

One sharp edge, stated rather than hidden: MLflow records each artifact's
location in the database as an **absolute path**. Posting the store between
runners works only because every job checks out to the same path on its runner.
Move the checkout and the database points at nothing. That is what "state" costs,
and it is why a real deployment puts the store on a server both jobs can reach
instead of posting it between them.

### Watching it refuse

Both refusals are reachable from the **Run workflow** button, because a gate you
have only read about is not a gate you believe in:

| Inputs | What happens |
|---|---|
| `data: data/auto-mpg-corrupt.csv` | `validate` exits 1, job 1 fails, **training never starts** |
| `seed_incumbent: true`, `min_improvement: 5` | `evaluate` exits 1, job 3 fails, **registration never runs** |

`seed_incumbent` is there because the registry starts empty in every run: the
first candidate has nothing to beat and is always promoted, so the gate's
interesting half is otherwise unreachable in CI. Set, it runs
train → evaluate → register once on the same data to put a version 1 in the
registry, and *then* trains the candidate the gate has to judge. A margin of 5
mean-squared-error is one no rerun of the same data can clear.

### Proving the refusals, on your own machine

A workflow you have to remember to trigger is a demonstration, not a guarantee.
`tests/test_pipeline_refusals.py` is the same two refusals as a test:

```bash
python -m pytest -m integration
```

It drives validate → train → evaluate → register as four real subprocesses,
stopping at the first non-zero exit exactly as a pipeline runner does, against a
throwaway SQLite store in a temporary directory - so it needs no credentials, no
network, and it cannot register anything into your own `mlflow.db`. It asserts
that corrupt data leaves the store empty because training never started, and
that a candidate the gate rejects leaves the registry holding exactly the
versions it held before - including when `register` is typed by hand afterwards,
which is the case the pipeline's job ordering cannot cover.

It asserts on exit codes and on the registry, never on the wording of a log
line, so improving a message does not break it.

### What the pull request shows

Each job writes its step's own output - the identical text you get in a terminal
- to the run's job summary, so the run's metrics and the gate's verdict are
readable from the pull request's checks without opening a log. The commands are
unchanged: the workflow types what the block under [What runs
today](#what-runs-today) tells you to type, and nothing in the entrypoints knows
it is running in CI.

No secrets, no accounts, no service connections, no third-party actions. Fork the
repository, open a pull request, and the pipeline runs.

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
#    the requirements manifest, which lives above it.
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
- **None of this repository's own code beyond `serving/`.** The pipeline's first
  stage wraps a function from `automobile.model_factory`, and cloudpickle stores
  that function as a *reference*, so the artifact does not unpickle unless that
  module is importable. This image used to copy the whole package in to satisfy
  that. It no longer needs to: the train step logs the model with `code_paths`,
  so the package travels *inside* the artifact and MLflow puts it on the path
  when it loads. The dependency is the model's, so the model carries it - which
  is also what lets the managed endpoint, which never sees this repository, load
  the same artifact.
- **`.dockerignore`** keeps the rest of the repository - your `.venv`, your
  `mlruns/`, the tests, the notebook - out of the build context entirely.

## The same container, on Kubernetes

`docker run` gets the service up. It does not keep it up, and it does not give
you a second copy when one is not enough. Those two properties - **résilience**
and **scalabilité** - are the reason an orchestrator exists, and `k8s/` is where
they are written down: a Deployment, a Service, three probes, resource requests
and limits, and a HorizontalPodAutoscaler.

Nothing in `k8s/` is generated and nothing in it is a chart. The Helm material
in `charts/` is a separate exercise - see [Deployment strategies](#deployment-strategies-blue-green-canary-and-shadow) -
and this is the plain-YAML layer underneath it, which is the layer worth being
able to read.

```
k8s/
  kind-cluster.yaml            The local cluster: one control plane, two workers.
  deployment.yaml              The pods, the probes, the requests and limits.
  service.yaml                 One stable address in front of N replicas.
  hpa.yaml                     Replicas follow CPU.
  metrics-server/              What the HPA needs, and does not get for free.
  labs/readiness-failure.yaml  Out of service, still running.
  labs/liveness-failure.yaml   Restarted.
  labs/loadgen.yaml            Something to scale in response to.
```

> **Delete the cluster when you stop working.** It does not stop on its own, it
> survives reboots, and it is invisible unless you go looking - there is no
> window to close and nothing in `docker ps` that reads as "a Kubernetes
> cluster". Idle, with one replica and metrics-server up, the three nodes hold
> **about 1.9 GB**:
>
> ```bash
> kind delete cluster --name ryvion   # the whole thing, in one command
> kind get clusters                   # "No kind clusters found."
> ```
>
> This is not tidiness. A later section of the course adds a monitoring stack to
> the same cluster, and on a 8 GB laptop a forgotten cluster from last week is
> the difference between that working and that swapping. Read [Teardown](#teardown)
> now rather than when you get to it.

### What you need

`kubectl` and [`kind`](https://kind.sigs.k8s.io/docs/user/quick-start/#installation),
on top of the container engine the previous section already needed. kind runs a
Kubernetes cluster as Docker containers, which is what makes this a laptop
exercise: no cloud account, no cluster to pay for, and teardown is one command.

```bash
# macOS / Linux
brew install kind kubectl
# Windows
winget install Kubernetes.kind Kubernetes.kubectl
# any platform, no package manager - a single binary
# https://kind.sigs.k8s.io/docs/user/quick-start/#installing-from-release-binaries
```

Docker Desktop and Rancher Desktop both ship a `kubectl`, so you may already
have one. `kubectl version --client` is the check.

### From an image on your laptop to a service in a cluster

```bash
# 0. Build the image first - the previous section, in full. The cluster deploys
#    the image you built; there is no registry in this story at all.
docker build -f serving/Dockerfile -t ryvion-mlops-serving:local .

# 1. Create the cluster. Two or three minutes the first time, while the node
#    image downloads; seconds afterwards.
kind create cluster --config k8s/kind-cluster.yaml
kubectl config use-context kind-ryvion
kubectl get nodes

# 2. Put the image INTO the cluster. This step is the one everybody forgets.
kind load docker-image ryvion-mlops-serving:local --name ryvion

# 3. Deploy.
kubectl apply -f k8s/deployment.yaml -f k8s/service.yaml -f k8s/hpa.yaml
kubectl rollout status deploy/ryvion-serving

# 4. Call it, from outside the cluster, on the port kind published.
curl -s localhost:30080/healthz
curl -s localhost:30080/readyz
curl -s -X POST localhost:30080/predict \
  -H 'Content-Type: application/json' \
  -d '{"records":[{"cylinders":8,"displacement":307.0,"horsepower":"130.0",
       "weight":3504,"acceleration":12.0,"model year":70,"origin":1,
       "car name":"chevrolet chevelle malibu"}]}'
# {"predictions":[14.975242297915628]}
```

**Step 2 is not optional and its absence does not look like its absence.** A
kind cluster's nodes are containers with their own image stores; they cannot see
your Docker daemon's images. Skip the load and the pods sit in `ErrImagePull`
while the kubelet tries to fetch `docker.io/library/ryvion-mlops-serving:local`
from a registry that has never heard of it. Rebuild the image and you must load
it again - the cluster keeps the copy it was given, so a stale deployment that
"ignores your changes" is almost always a forgotten `kind load`.

The Deployment sets `imagePullPolicy: IfNotPresent` for the same reason. With
`Always`, the kubelet would go to a registry even for an image it already has.

### Résilience: the two probes do different things

There are three probes in `k8s/deployment.yaml` and the difference between two
of them is the whole lesson.

| Probe | Points at | On failure |
|---|---|---|
| `startupProbe` | `/healthz` | Nothing yet - it suspends liveness while the model loads. |
| `livenessProbe` | `/healthz` | **Restarts the container.** |
| `readinessProbe` | `/readyz` | **Removes the pod from the Service. Does not restart it.** |

`/healthz` deliberately knows nothing about the model, and `/readyz` is the one
that reports it. That pairing is not decoration. A missing model is not fixed by
a restart - the container comes back and fails identically - so it must fail
*readiness*, which routes traffic away and leaves the container running for you
to interrogate. A wedged process is not fixed by routing around it, so it must
fail *liveness*, which restarts it.

Both behaviours are in `k8s/labs/`, and both are worth causing on purpose.

**Readiness failure - out of service, still running.** The same image, told to
load a model from a path that has none:

```bash
kubectl apply -f k8s/labs/readiness-failure.yaml
kubectl get pods -l lab=readiness-failure -w
# NAME                              READY   STATUS    RESTARTS   AGE
# ryvion-nomodel-7469f67d6d-qfnx8   0/1     Running   0          2m22s
```

`Running`, `0/1`, and `RESTARTS 0` - and it stays that way for as long as you
watch. The pod is in the EndpointSlice as `ready: false`, so kube-proxy sends it
nothing:

```bash
kubectl get endpointslices -l kubernetes.io/service-name=ryvion-nomodel -o yaml | grep -A4 addresses
#   - addresses:
#     - 10.244.2.7
#     conditions:
#       ready: false
#       serving: false
```

The container is healthy; only the model is missing, and because the container
was left alive you can ask it so:

```bash
kubectl exec deploy/ryvion-nomodel -- python -c \
  "import urllib.request;print(urllib.request.urlopen('http://127.0.0.1:8000/healthz').read())"
# b'{"status":"ok"}'
```

while `/readyz` answers `503` with `could not load a model from
/tmp/there-is-no-model-here`. Note what `kubectl describe pod` gives you and
what it does not: `Readiness probe failed: HTTP probe failed with statuscode:
503`, the status code and nothing else. The reason lives in the response body,
and you can only go and get it because nothing restarted the pod.

```bash
kubectl delete -f k8s/labs/readiness-failure.yaml
```

**Liveness failure - restarted, over and over.** This one patches the real
Deployment so that the liveness probe points at a path that does not exist:

```bash
kubectl patch deployment ryvion-serving --patch-file k8s/labs/liveness-failure.yaml
kubectl get pods -l app.kubernetes.io/name=ryvion-serving -w
# ryvion-serving-5df4745c65-tnqpw   1/1   Running            0             18s
# ryvion-serving-5df4745c65-tnqpw   0/1   Running            1 (2s ago)    33s
# ryvion-serving-5df4745c65-tnqpw   0/1   Running            2 (2s ago)    63s
# ryvion-serving-5df4745c65-tnqpw   0/1   Running            3 (2s ago)    93s
# ryvion-serving-5df4745c65-tnqpw   0/1   Running            4 (3s ago)    2m4s
# ryvion-serving-5df4745c65-tnqpw   0/1   CrashLoopBackOff   4 (3s ago)    2m34s

kubectl describe pod -l app.kubernetes.io/name=ryvion-serving | grep -E "Unhealthy|Killing"
# Warning  Unhealthy  Liveness probe failed: HTTP probe failed with statuscode: 404
# Normal   Killing    Container serving failed liveness probe, will be restarted

kubectl apply -f k8s/deployment.yaml    # undo it
```

The application never crashed. `/healthz` answered `200` the entire time; the
*probe* was wrong, and a wrong liveness probe restarts a working service until
it is declared to be in a crash loop. That failure mode looks exactly like a
broken application in every dashboard you will ever see, which is the argument
for keeping `/healthz` free of anything a restart cannot fix.

Note also the flicker in the `READY` column. Every restart takes the pod out of
the Service for as long as it takes to come back, so self-healing is not free on
a single replica - which is the argument for the next section.

### Requests and limits

```yaml
resources:
  requests: { cpu: 100m, memory: 256Mi }
  limits:   { cpu: 500m, memory: 512Mi }
```

Two different words for two different mechanisms, and the asymmetry between the
CPU pair and the memory pair is deliberate:

- A **request** is what the *scheduler* reserves. A node needs this much
  unclaimed before a replica can land on it. It is also the denominator the
  autoscaler divides by - a Deployment with no CPU request cannot be autoscaled
  on CPU at all, and the HPA will report `<unknown>/60%` forever.
- A **limit** is what the *kernel* enforces. CPU over the limit is throttled;
  memory over the limit is an OOM kill. So CPU is set generously above its
  request - a busy pod is allowed to burst into idle capacity, and that burst is
  exactly what the autoscaler measures - while memory sits close to what the
  process actually uses. `kubectl top pods` reports an idle replica at about
  `175Mi`, which is where `256Mi` and `512Mi` came from rather than from
  a round number that looked safe.

### Scalabilité: autoscaling, and the part nobody warns you about

Apply the HPA on a fresh kind cluster and it does nothing:

```bash
kubectl get hpa
# NAME             REFERENCE                   TARGETS              MINPODS   MAXPODS   REPLICAS
# ryvion-serving   Deployment/ryvion-serving   cpu: <unknown>/60%   1         6         1
kubectl top pods
# error: Metrics API not available
```

**A HorizontalPodAutoscaler cannot see CPU by itself.** It reads the metrics API
(`metrics.k8s.io`), and something has to serve that API. Managed clusters have
`metrics-server` pre-installed and never mention it; kind does not have it at
all. There is no error - just `<unknown>`, indefinitely.

Installing it is one command, and then there is a second trap behind the first:

```bash
kubectl apply -f https://github.com/kubernetes-sigs/metrics-server/releases/download/v0.9.0/components.yaml
kubectl -n kube-system logs deploy/metrics-server | grep scrape
# E0821 15:08:51 scraper.go:149] "Failed to scrape node" err="Get
# \"https://172.24.0.3:10250/metrics/resource\": tls: failed to verify certificate:
# x509: cannot validate certificate for 172.24.0.3 because it doesn't contain any
# IP SANs" node="ryvion-worker"
```

metrics-server verifies the certificate each kubelet presents. kind's kubelets
serve **self-signed** certificates that no cluster CA has signed, so every
scrape fails, the pod sits at `0/1 Ready` with `no metrics to serve`, and
`kubectl top` still errors. Use the kustomization in this repository instead,
which pins the version and patches in the one flag that resolves it:

```bash
kubectl apply -k k8s/metrics-server
kubectl -n kube-system rollout status deploy/metrics-server
kubectl top pods
# NAME                              CPU(cores)   MEMORY(bytes)
# ryvion-serving-5b9fbbf6c9-hgq7l   2m           174Mi
```

The flag is `--kubelet-insecure-tls`, it is patched in visibly rather than
hidden behind an install switch, and `k8s/metrics-server/kustomization.yaml`
spells out what it costs: metrics-server will then believe anything answering on
the kubelet port. It is the documented answer for kind and the wrong answer in
production, where the fix is to let the cluster CA sign kubelet serving
certificates.

**Now drive load.** `k8s/labs/loadgen.yaml` runs four clients posting batches at
the Service - using the serving image itself, because it is already on the nodes
and a lab that fails at `ImagePullBackOff` teaches nothing.

```bash
kubectl apply -f k8s/labs/loadgen.yaml

# In two more terminals - the HPA's own view, and the pods it is creating:
kubectl get hpa ryvion-serving -w
kubectl get pods -l app.kubernetes.io/name=ryvion-serving -w
```

The transcripts below are those two watches interleaved: the HPA's
`TARGETS MINPODS MAXPODS REPLICAS` columns, then how many of the pods that
exist are Ready.

```
17:35:06  cpu: 35%/60%    1  6  1     ready=1/1
17:35:16  cpu: 87%/60%    1  6  1     ready=1/2
17:35:27  cpu: 501%/60%   1  6  2     ready=2/4
17:35:48  cpu: 501%/60%   1  6  4     ready=4/6
17:35:58  cpu: 421%/60%   1  6  6     ready=6/6
```

One replica pinned at its 500m limit is 500% of its 100m request, so
`desired = ceil(1 x 500/60) = 9`, capped at `maxReplicas: 6`. The climb is
`1 -> 2 -> 4 -> 6` because the scale-up policy allows +100% every 15 seconds.
Note the lag at the start: metrics are collected every 15s and the HPA
reconciles every 15s, so load takes about half a minute to become replicas. An
autoscaler is a capacity tool, not a latency tool.

The new replicas land on both workers, which is what the two-worker cluster and
the `topologySpreadConstraints` in `deployment.yaml` are for - a service whose
six replicas share one node has not bought much résilience:

```bash
kubectl get pods -l app.kubernetes.io/name=ryvion-serving -o wide
# ryvion-serving-...-2kk4h   1/1   Running   10.244.1.7   ryvion-worker2
# ryvion-serving-...-bhsrm   1/1   Running   10.244.2.5   ryvion-worker
# ryvion-serving-...-jc4q6   1/1   Running   10.244.1.5   ryvion-worker2
# ryvion-serving-...-vb8vg   1/1   Running   10.244.2.6   ryvion-worker
```

**Then take the load away**, and watch the asymmetry:

```bash
kubectl delete -f k8s/labs/loadgen.yaml
```

```
17:38:49  cpu: 162%/60%   1  6  6     ready=6/6
17:38:59  cpu: 2%/60%     1  6  6     ready=6/6
17:39:41  cpu: 2%/60%     1  6  6     ready=3/3
17:40:02  cpu: 2%/60%     1  6  3     ready=3/3
17:40:33  cpu: 2%/60%     1  6  1     ready=1/1
```

CPU collapses immediately; replicas do not. `scaleDown.stabilizationWindowSeconds`
holds the highest recent recommendation for 60 seconds first, so a lull cannot
throw away capacity that a spike is about to need again, and then the
50%-per-30s policy steps down `6 -> 3 -> 1` rather than dropping straight to the
floor. The upstream default for that window is **300 seconds**; `k8s/hpa.yaml`
shortens it to 60 so the second half of this demonstration fits inside a class,
and says so in a comment.

`kubectl describe hpa ryvion-serving` has the whole story in its events:

```
Normal  SuccessfulRescale  15m  New size: 2; reason: cpu resource utilization (percentage of request) above target
Normal  SuccessfulRescale  14m  New size: 4; reason: cpu resource utilization (percentage of request) above target
Normal  SuccessfulRescale  14m  New size: 6; reason: cpu resource utilization (percentage of request) above target
Normal  SuccessfulRescale  10m  New size: 3; reason: All metrics below target
Normal  SuccessfulRescale  10m  New size: 1; reason: All metrics below target
```

One thing the HPA does *not* do: it will not undo `replicas:` in
`deployment.yaml`. Both fields own the replica count, and whichever was written
last wins - which is why re-applying the Deployment during a scale-up snaps it
back to 1 and the autoscaler then has to climb again.

### Teardown

**Do this every time you stop working, not once at the end of the course.**

A kind cluster is three long-running Docker containers. Nothing stops them: not
closing your terminal, not logging out, not rebooting - the Docker daemon starts
them again. They do not appear in any menu, and in `docker ps` they look like
three ordinary containers with unmemorable names. The only thing that tells you
one is running is asking:

```bash
kind get clusters
```

Measured on the cluster this section builds - three nodes, one serving replica,
metrics-server - the resident cost is about **1.9 GB**:

```
ryvion-control-plane   967.1MiB
ryvion-worker          565.4MiB
ryvion-worker2         388.7MiB
```

During the autoscaling lab, with six replicas and four load generators, it is
considerably more. Deleting the cluster takes all of it back, along with the
Deployment, the Service, the HPA, metrics-server and the loaded image:

```bash
kind delete cluster --name ryvion
kind get clusters
# No kind clusters found.
```

Run the second command. `kind delete cluster` without `--name` deletes a cluster
called `kind`, not this one, and reports success while `ryvion` keeps running -
which is the single easiest way to believe you have cleaned up when you have
not.

The built image stays in your local Docker daemon, which is usually what you
want between sessions. `docker rmi ryvion-mlops-serving:local` reclaims its
1.4 GB when you are finished with it for good.

Why this section is emphatic: a later part of the course installs Prometheus and
Grafana onto this same cluster, and that is the memory ceiling of the whole
module on a 8 GB laptop. A cluster left running from a previous session is the
usual reason it does not fit.

## Deployment strategies: blue-green, canary and shadow

Deploying a new model is not one decision, it is a choice between three, and the
course examines all three by name. `charts/` is where they live.

| Strategy | What happens | What it buys |
|---|---|---|
| **Blue-green** | Both versions are up and healthy; all traffic goes to one of them. The cutover is a configuration change, and so is the way back. | A release you can reverse in seconds, without a rebuild and without waiting for pods to start. |
| **Canary** | A defined percentage of traffic reaches the new version while the rest stays on the old one. The percentage is adjustable while both are live. | Finding out whether the new model is worse *on real traffic*, at a blast radius you chose. |
| **Shadow** | The new version receives a **copy** of every request. Its predictions are recorded. The caller never sees them. | Comparing a candidate against production traffic at zero risk - because there is no path at all from it back to a caller. |

The third one is why this block is worth doing locally. A managed endpoint will
split traffic for you; what it will not do is run a model against production
traffic while guaranteeing that nothing it produces is ever returned. Shadow
deployment is the strategy that most needs the traffic layer to be something you
can open and read.

### The charts, and what was wrong with them

Two of these charts are inherited. They have been in this repository for five
years, they are on the slides, and they had never been installed once. That is
not a guess. Check out the chart as it was inherited and it does not survive
`helm template`, let alone `helm install`:

```
Error: abtest-model/templates/deployment.yaml:23:26
  executing "abtest-model/templates/deployment.yaml" at <.Values.deployment.image.name>:
    nil pointer evaluating interface {}.name
```

```
charts/
  abtest-model/         One model version, deployed. Installed twice, once per colour.
  abtest-router/        The traffic layer. Blue-green, canary and shadow, no mesh.
    files/router.py     ~300 lines of standard library. Mounted from a ConfigMap.
  abtest-istio/         The same three strategies as a Gateway and a VirtualService.
  load_test.sh          Sends N predictions and tallies which version answered.
```

| Where | Was | Is | Why it mattered |
|---|---|---|---|
| `abtest-model` values | no `deployment.image.name` at all | set, per colour | The template read it. Every install failed. |
| `abtest-model` ports | `5001`, plus a `probe` port `8086` | `8000`, and no second port | The serving container listens on 8000 and has no probe port. The Service pointed at nothing. |
| `abtest-model` probes | none | `/healthz`, `/readyz`, and a startup probe | Without readiness, a cutover sends traffic to a pod that has not finished unpickling its model. |
| `abtest-model` resources | none | requests and limits | Nothing could be scheduled predictably; no HPA could ever have seen these pods. |
| `abtest-istio` API | `networking.istio.io/v1alpha3` | `networking.istio.io/v1` | v1alpha3 was removed in Istio 1.22. |
| `abtest-istio` path | `/score` | `/predict` | `/score` is not an endpoint of this service. |
| `abtest-istio` mirror | absent | `mirror` + `mirrorPercentage` | The chart advertised three strategies and implemented two. |
| both charts | one model, deployed against a byte-identical copy of itself | two models trained with different split parameters | An A/B test between identical artifacts tests the load balancer. |
| `load_test.sh` | a bare `GET` with a header, one every 200 ms, untallied | `POST /predict`, tallied by version | 200 requests took 40 seconds and told you nothing. |

The last row of that table is the important one, and it is the reason this slice
exists at all.

### Two model versions that genuinely differ

There is no CI in this loop and none is needed: train twice with different split
parameters and you have two measurably different models.

```bash
python -m automobile.entrypoints.train --run-name blue
python -m automobile.entrypoints.train --run-name green --random-seed 7 --test-size 0.35
```

`blue` is the repository's default run - seed 42, 20% held back - and `green`
holds back 35% of a differently-shuffled split, so it fits on 258 rows instead of
318 and lands somewhere else.

```
                        blue            green
test r²                 0.8476          0.7669
test rmse               2.8628          3.5292
cylinders coefficient   -0.260          -1.655
displacement            1.442           2.538
weight                  -5.684          -5.489

mean |blue - green| over all 398 rows: 0.3965 mpg
max  |blue - green|:                   2.0756 mpg
rows differing by more than 0.25 mpg:  232 / 398
```

Same request, two answers, and the answers are how you tell which version
replied:

```
blue   {"predictions":[14.975242297915628]}
green  {"predictions":[14.358430890938523]}
```

The model is a build input, so a version of the model is a version of the image.
Export both and build both - the second build reuses every layer but the last,
so it takes seconds:

```bash
mlflow artifacts download --artifact-uri models:/<blue-id>  --dst-path serving/model-blue
mlflow artifacts download --artifact-uri models:/<green-id> --dst-path serving/model-green

docker build -f serving/Dockerfile --build-arg MODEL_DIR=serving/model-blue \
  -t ryvion-mlops-serving:blue .
docker build -f serving/Dockerfile --build-arg MODEL_DIR=serving/model-green \
  -t ryvion-mlops-serving:green .
```

`MODEL_DIR` was already a build argument of `serving/Dockerfile`. Nothing about
the image had to change to make it serve two models; that is what a build
argument is for.

### Setting the lab up

```bash
# 0. A cluster, and the images inside it.
kind create cluster --config k8s/kind-cluster.yaml
kubectl config use-context kind-ryvion

kind load docker-image ryvion-mlops-serving:blue  --name ryvion
kind load docker-image ryvion-mlops-serving:green --name ryvion

# The router runs a stock Python image with a script mounted into it, so that
# image has to be in the cluster too.
docker pull python:3.11.16-slim-bookworm
docker save --platform linux/amd64 python:3.11.16-slim-bookworm -o /tmp/py311.tar
kind load image-archive /tmp/py311.tar --name ryvion

# 1. Both model versions, from the same chart.
helm install model-blue charts/abtest-model -n abtesting --create-namespace \
  --set deployment.name=model-blue --set deployment.bluegreen=blue \
  --set deployment.image.name=ryvion-mlops-serving:blue

helm install model-green charts/abtest-model -n abtesting \
  --set deployment.name=model-green --set deployment.bluegreen=green \
  --set deployment.image.name=ryvion-mlops-serving:green

# 2. The traffic layer, published on the port kind maps to the laptop.
helm install abtest-router charts/abtest-router -n abtesting \
  --set strategy=bluegreen --set live=blue

kubectl -n abtesting rollout status deploy/model-blue
kubectl -n abtesting rollout status deploy/model-green
kubectl -n abtesting rollout status deploy/abtest-router
```

Two notes on the `docker save --platform` line, because both cost time to find.
`kind load docker-image python:3.11.16-slim-bookworm` fails outright with
`ctr: content digest sha256:...: not found`: Docker's containerd image store keeps
a multi-platform index for a pulled image, kind imports it with `--all-platforms`,
and the blobs for the platforms you did not pull are simply absent.
`docker save --platform linux/amd64` exports one platform, and
`kind load image-archive` then has everything it refers to. Locally *built*
images do not hit this, which is why the two serving images load without ceremony.

The router's Service claims **NodePort 30080**, the same port `k8s/service.yaml`
claims. Deploy both and the second fails with `provided port is already
allocated`. Delete the plain-YAML deployment first, or set
`--set gateway.nodePort=` to something else and reach the router with
`kubectl port-forward` instead.

`model-svc-blue` and `model-svc-green` are `ClusterIP`. The router is the only
thing published, and that is deliberate: it is what makes "the shadow is never
served" a statement about reachability rather than about intent.

### Blue-green, and the rollback

One release, upgraded in place. Revision 1 serves blue:

```
$ curl -s localhost:30080/_router/status
{"strategy": "bluegreen", "serving": [{"name": "blue", "url": "http://model-svc-blue...:8000",
 "weight": 100}], "mirroring": null, "requests": {"served": {}, "mirrored": {}}}

$ curl -s -D - localhost:30080/predict -H 'Content-Type: application/json' -d @record.json
X-Model-Version: blue
{"predictions":[14.975242297915628]}
```

The cutover:

```
$ helm upgrade abtest-router charts/abtest-router -n abtesting \
    --set strategy=bluegreen --set live=green
STATUS: deployed
REVISION: 2

X-Model-Version: green
{"predictions":[14.358430890938523]}
```

And the part that matters more than the cutover - **the way back**:

```
$ helm history abtest-router -n abtesting
REVISION  UPDATED                   STATUS      CHART                DESCRIPTION
1         Fri Aug 21 21:49:38 2026  superseded  abtest-router-0.1.0  Install complete
2         Fri Aug 21 21:50:19 2026  deployed    abtest-router-0.1.0  Upgrade complete

$ helm rollback abtest-router 1 -n abtesting
Rollback was a success! Happy Helming!

X-Model-Version: blue
{"predictions":[14.975242297915628]}

$ helm history abtest-router -n abtesting
REVISION  UPDATED                   STATUS      CHART                DESCRIPTION
1         Fri Aug 21 21:49:38 2026  superseded  abtest-router-0.1.0  Install complete
2         Fri Aug 21 21:50:19 2026  superseded  abtest-router-0.1.0  Upgrade complete
3         Fri Aug 21 21:50:34 2026  deployed    abtest-router-0.1.0  Rollback to 1
```

Read revision 3's description. `helm rollback` is not a second forward change
that happens to restore the old values - it is recorded as a rollback, and the
release history says which revision it went back to. That distinction is the
whole reason to keep the traffic decision in a Helm release instead of in
`kubectl edit svc`.

Notice what did **not** happen: neither model pod restarted, and nothing was
rebuilt. Blue and green were both up and healthy the entire time. That is the
property being bought - the rollback costs one pod restart of a 20 MB router,
not a redeploy of a model.

### Canary, and adjusting the split

```
$ helm upgrade abtest-router charts/abtest-router -n abtesting \
    --set strategy=canary --set canary.weight=10
REVISION: 4

$ charts/load_test.sh 200
POST http://localhost:30080/predict x200
    180 blue
     20 green
blue       180   90.0%
green       20   10.0%
```

Then move it, with no model pod touched:

```
$ kubectl -n abtesting get pods -l app=model
model-blue-c5dbf99c9-l427j     age: 2m22s
model-green-69c46f8dbc-qdcmn   age: 2m19s

$ helm upgrade abtest-router charts/abtest-router -n abtesting \
    --set strategy=canary --set canary.weight=40
REVISION: 5

$ charts/load_test.sh 200
POST http://localhost:30080/predict x200
    120 blue
     80 green
blue       120   60.0%
green       80   40.0%

$ kubectl -n abtesting get pods -l app=model
model-blue-c5dbf99c9-l427j     age: 3m19s   # same pods, same age
model-green-69c46f8dbc-qdcmn   age: 3m16s
```

Exactly 120 and 80, not "about". The router splits with **smooth weighted
round-robin**, the algorithm nginx uses for upstream weights: each version
accumulates its weight on every request, the largest accumulator wins and then
pays the total back. Over any run of 100 requests each version is picked exactly
`weight` times, and the picks are interleaved rather than arriving in blocks.

A mesh usually splits on a hash of the request instead, so its numbers wobble -
the same 200 requests through Istio at 25% gave 51 rather than 50 further down
this page. Neither is more correct; it is worth saying out loud when the counts
come out suspiciously round.

**One honest wrinkle.** The first run of the 40% test reported 122/78, not
120/80, and `/_router/status` explained why: the new router pod had only counted
196 of the 200 requests. Four had been answered by the *outgoing* pod, which was
still in the Service's EndpointSlice for a moment after `kubectl rollout status`
returned, and was still on 90/10. A traffic split is a property of a running
process, and for a second during a rollout there are two of them. Send the 200
requests again once the dust settles and the numbers are exact.

### Shadow: mirrored, recorded, never served

```
$ helm upgrade abtest-router charts/abtest-router -n abtesting \
    --set strategy=shadow --set shadow.serving=blue --set shadow.mirror=green
REVISION: 6

$ curl -s localhost:30080/_router/status
{"strategy": "shadow",
 "serving":   [{"name": "blue",  "url": "...model-svc-blue...",  "weight": 100}],
 "mirroring":  {"name": "green", "url": "...model-svc-green...", "percentage": 100,
                "paths": ["/predict"], "recorded": 0},
 "requests": {"served": {}, "mirrored": {}}}
```

`serving` and `mirroring` are two different fields, and that is the whole design.
The mirror target is not in the routing table, so there is no code path that can
select it to answer a caller.

100 requests through the gateway:

```
distinct X-Model-Version values seen by the caller:
    100 blue
distinct response bodies seen by the caller:
    100 {"predictions":[14.975242297915628]}
```

One hundred responses, one distinct body, and it is blue's. Green's answer to the
same record - asked of `model-svc-green` directly, from inside the cluster - is:

```
{"predictions":[14.358430890938523]}
```

That number appears in no response the caller received. It appears here:

```
$ curl -s localhost:30080/_router/shadow
mirroring: green
recorded: 100
{"at": 1787342127.088801, "path": "/predict", "served_by": "blue",
 "served_predictions": [14.975242297915628],
 "shadow": "green", "shadow_status": 200,
 "shadow_predictions": [14.358430890938523]}
```

and in the router's log, one JSON object per mirrored request, which is what a
real shadow deployment would ship to the same place its metrics go:

```
$ kubectl -n abtesting logs deploy/abtest-router --tail=1
{"ts": 1787342127.6684906, "msg": "mirrored", "path": "/predict", "served_by": "blue",
 "served_predictions": [14.975242297915628], "shadow": "green",
 "shadow_status": 200, "shadow_predictions": [14.358430890938523]}
```

Both predictions are recorded side by side on purpose. A shadow deployment whose
record does not include what production said at the same moment tells you what
the candidate predicted but not whether it disagreed.

Now try to get the shadow's answer out of it. `x-api-version` pins a request to a
named version, and it is the header `charts/load_test.sh` has always sent:

```
$ curl -si localhost:30080/predict -H 'x-api-version: green' -d @record.json
HTTP/1.1 409 Conflict
{"detail": "'green' is a shadow deployment: it receives mirrored traffic and its
 predictions are recorded, never served. See GET /_router/shadow."}
```

The counters agree:

```
"requests": {"served": {"blue": 100}, "mirrored": {"green": 100}}
```

Green answered 100 requests and served zero of them.

### The memory decision: a mesh, or 300 lines of Python

The inherited chart implements all of this with Istio, and Istio is the right
answer on a cluster that already runs one. The question for this course is
narrower: what does it cost on the laptop that also has to fit a monitoring
stack, and is the difference worth what it buys?

That is a measurement, not an opinion. All four numbers below are
`docker stats --no-stream` against the three kind node containers, on the same
cluster, in one sitting, on an 8 GB machine (`7.611GiB` total as Docker reports
it):

| Cluster state | control-plane | worker | worker2 | total | delta |
|---|---|---|---|---|---|
| Fresh 3-node cluster, nothing deployed | 659.5 MiB | 140.1 MiB | 145.0 MiB | **945 MiB** | - |
| + metrics-server, both model versions, the router | 1046.5 MiB | 656.5 MiB | 554.3 MiB | **2257 MiB** | +1312 MiB |
| + istiod and an Istio ingress gateway | 1147.9 MiB | 709.5 MiB | 622.4 MiB | **2480 MiB** | +222 MiB |
| + a sidecar injected into each model pod | 1191.9 MiB | 1015.0 MiB | 442.8 MiB | **2650 MiB** | +392 MiB total |

And per pod, from `kubectl top`:

```
abtest-router                18Mi
model-blue / model-green    206Mi each   (182Mi + a 27Mi istio-proxy once injected)
istiod                       41Mi
istio-ingressgateway         25Mi
```

**Istio, on this cluster, costs about 390 MiB resident - not the ~1 GB the plan
for this slice assumed.** The estimate was pessimistic and the measurement says
so. A mesh is affordable here; that is the honest finding and it is worth
recording, because the decision below does not rest on it.

The router is **18 MiB**. That is the comparison that survives: not 1 GB against
20 MB, but 390 MiB against 18 MiB, twenty times cheaper, on the machine where a
later section of this course adds Prometheus and Grafana to the same cluster.
Alongside that:

- Istio is three more Helm releases, about thirty CRDs and roughly 600 MB of
  images to pull. None of it works offline the first time. The router needs one
  image the serving build has already pulled.
- Turning the mesh on means labelling the namespace and **restarting every
  workload** so a sidecar can be injected. That is a real operation with real
  consequences, and it is not one you want between two five-minute labs.
- Istio's `mirror` sends the copy and **discards the response**. The mirrored
  workload's own logs record that a request arrived; nothing records what it
  predicted. "Its predictions are recorded" - the criterion this slice is judged
  on - needs machinery Istio does not supply. The router records both predictions
  because recording them is its entire reason for existing.

So the router is the default and the mesh is the reference. The teaching goal is
that a student can operate all three strategies and explain the difference; it is
not that they install a service mesh.

### The Istio path, for a cluster that already has a mesh

`charts/abtest-istio` speaks the same vocabulary as `charts/abtest-router` -
`strategy`, `live`, `canary.weight`, `shadow.serving`, `shadow.mirror` - so the
two are diffable as the same idea in two languages. Everything below was run
against a real Istio 1.30.3 on the cluster measured above.

```bash
helm repo add istio https://istio-release.storage.googleapis.com/charts
helm install istio-base istio/base --version 1.30.3 -n istio-system --create-namespace
helm install istiod istio/istiod --version 1.30.3 -n istio-system --wait
helm install istio-ingressgateway istio/gateway --version 1.30.3 -n istio-system --wait

helm install abtest-istio charts/abtest-istio -n abtesting \
  --set strategy=shadow --set shadow.serving=blue --set shadow.mirror=green

# kind has no cloud load balancer, so the gateway Service stays <pending>.
kubectl -n istio-system port-forward svc/istio-ingressgateway 18080:80
```

Shadow, through the mesh:

```
access-log POST /predict counts before: blue=0 green=0

--- 20 requests through the Istio ingress gateway ---
distinct bodies the caller received:
     20 {"predictions":[14.975242297915628]}

access-log POST /predict counts after:  blue=20 green=20
delta: blue=20 green=20
```

Green handled twenty requests and served none of them - and note how that had to
be established: by counting lines in the mirrored pod's *access log*, because
Envoy threw the predictions away. The router's `/_router/shadow` is what that
gap costs you.

Asking Istio for the shadow by name is refused the same way, by a
`directResponse` rule placed ahead of the routing rules:

```
HTTP/1.1 409 Conflict
{"detail":"green is a shadow deployment: it receives mirrored traffic and its
 predictions are recorded, never served."}
```

Canary and blue-green, through the mesh:

```
$ helm upgrade abtest-istio charts/abtest-istio -n abtesting \
    --set strategy=canary --set canary.weight=25
     51 {"predictions":[14.358430890938523]}    # green, 25.5%
    149 {"predictions":[14.975242297915628]}    # blue

$ helm upgrade abtest-istio ... --set strategy=bluegreen --set live=blue    # rev 3
{"predictions":[14.975242297915628]}
$ helm upgrade abtest-istio ... --set strategy=bluegreen --set live=green   # rev 4
{"predictions":[14.358430890938523]}
$ helm rollback abtest-istio -n abtesting                                   # rev 5
{"predictions":[14.975242297915628]}
```

Note 51 rather than 50: Istio distributes per request rather than in a fixed
rotation, so its splits are approached rather than hit.

### What the router is not

It is a teaching artifact for one job, and everything a mesh does beyond that job
is absent from it. Written down, so that nobody mistakes it for the real thing:

- **No mutual TLS, no identity, no authorization policy.** Plain HTTP inside the
  cluster. A mesh's main selling point in production is the one thing this does
  not attempt.
- **No retries, timeouts beyond a flat one, circuit breaking or outlier
  detection.** A failing upstream stays in the rotation.
- **No telemetry.** Counters at `/_router/status` and a log line per mirrored
  request, held in memory and lost when the pod restarts. No metrics endpoint,
  and nothing for issue #24's Prometheus to scrape yet.
- **It is a single hop, not a data plane.** One Deployment in front of two
  Services. It sees only traffic that comes through the gateway; service-to-service
  calls inside the cluster bypass it entirely, where a sidecar mesh would not.
- **One replica by default, so it is a single point of failure**, and the split
  counters are per-pod - scale it to two and each keeps its own rotation.
- **Header-pinned requests do not advance the split.** Convenient for testing,
  and a difference from a mesh worth knowing before you trust a tally.

If any of those matter, the answer is `charts/abtest-istio` and the 390 MiB, not
a bigger `router.py`.

### Teardown

The same rule as every other cluster in this course, and the same trap:

```bash
kind delete cluster --name ryvion
kind get clusters
# No kind clusters found.
```

Run the second command. `kind delete cluster` without `--name` deletes a cluster
called `kind`, reports success, and leaves `ryvion` running.

If you installed Istio and want the cluster back without it,
`helm uninstall istio-ingressgateway istiod istio-base -n istio-system`, then
`kubectl label namespace abtesting istio-injection-` and restart the model
deployments to shed the sidecars. Deleting the cluster is faster.

## Repository layout

```
automobile/            The domain package.
  entrypoints/         One argparse shell per pipeline step. No domain logic.
serving/               The hand-built serving application and its Dockerfile.
k8s/                   Plain Kubernetes manifests, and the probe/autoscaling labs.
data/                  The seed dataset, committed. 398 rows, six of them defective,
                       plus a deliberately corrupt copy for the contract to refuse.
environments/          The three dependency manifests, and the lockfile.
tests/                 Unit tests, plus one integration test of the refusal paths
                       behind `-m integration`. No credentials, no network.
.github/workflows/     GitHub Actions: the always-on quality gate, and the pipeline.
.github/actions/       One local composite action: Python and the locked environment.
.pipelines/            Azure Pipelines: the definitions the course studies.
charts/                Helm charts: two model versions, and the traffic layer that
                       does blue-green, canary and shadow deployment on top of them.
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

- **`.github/workflows/ci.yml`** is the always-on gate on the *code*. It runs
  lint, the unit tests, and then the integration test that defends the pipeline's
  refusal paths, on every pull request, and needs no setup at all: fork the
  repository, open a pull request, and it runs. This is what gives coursework
  instant feedback.
- **`.github/workflows/pipeline.yml`** is the gate on the *model*: the four
  pipeline steps as four dependent jobs, described under [The pipeline, as a CI
  workflow](#the-pipeline-as-a-ci-workflow). Same triggers, same absence of
  setup.
- **`.pipelines/pr.yml`** is the same gate on the CI platform the module
  teaches, connected to a student's own organisation and their own fork. It is
  the artifact to read and extend, and it makes the split between code host and
  CI platform - two definitions, two systems, one repository - concrete.

Both run `flake8 .` and `pytest`. In the predecessor, the template holding those
two commands was commented out of the pipeline, so neither ever ran. Here it is
included.

## Carried over from the predecessor

The Helm install and upgrade templates (`.pipelines/helm-*.yml`) and the
exploratory notebook (`notebooks/`) are copied across byte-for-byte.

The Helm charts (`charts/abtest-model`, `charts/abtest-istio`) and the load
generator (`charts/load_test.sh`) were carried across byte-for-byte too, and are
no longer: see [Deployment strategies](#deployment-strategies-blue-green-canary-and-shadow)
for what was wrong with each of them and what it was repaired to. They were
repaired rather than replaced, because they are on the slides.

Everything else from that repository was deleted by design: the orchestration
package, the duplicated scoring scripts, the R and Databricks training path, and
the parallel batch-scoring path.

## Where this is going

`docs/PRD-workstream-0-sdk-v2-rebuild.md` is the plan in full, and the open
issues are its slices: local training to a signed model artifact, the data
contract, the quality gate, the serving container, then the cloud pipeline,
managed endpoint and CI/CD definitions.
