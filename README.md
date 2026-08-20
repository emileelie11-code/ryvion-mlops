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

Nothing else. Everything in this slice runs on a laptop with no cloud account,
no credentials and no container engine.

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

The last command should end in `24 passed`. If it does, your environment matches
the one the pull-request gate uses.

## What runs today

The four pipeline steps exist as command-line entrypoints, installed by step 3
above:

| Command | Module | Step |
|---|---|---|
| `automobile-validate` | `automobile.entrypoints.validate` | Check the data against its contract |
| `automobile-train` | `automobile.entrypoints.train` | Fit the model pipeline and log the artifact |
| `automobile-evaluate` | `automobile.entrypoints.evaluate` | Apply the promote-or-reject gate |
| `automobile-register` | `automobile.entrypoints.register` | Register the promoted model |

Each currently parses its arguments and then raises `NotImplementedError`: this
slice establishes the toolchain, the packaging, the dependency pins and the CI
gate, and the steps are filled in by the slices that follow. `--help` works on
all four, and the test suite holds that surface in place while the bodies
arrive.

## Repository layout

```
automobile/            The domain package.
  entrypoints/         One argparse shell per pipeline step. No domain logic.
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
you and CI actually install. It pins all 149 direct and transitive packages,
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
