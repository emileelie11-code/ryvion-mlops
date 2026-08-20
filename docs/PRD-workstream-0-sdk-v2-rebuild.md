# PRD — Workstream 0: Ryvion MLOps teaching repo, rebuilt on Azure ML SDK v2

**Status:** ready for implementation
**Scope:** this repository only
**Blocks:** all slide production for the course sections on CI/CD, MLOps pipelines, and testing/observability

---

## Problem Statement

This repository is the worked example for a 35-hour graduate MLOps module. Three of the six
course sections are taught *out of it* — their slides, screenshots and step-by-step exercises are
all captures of this code running. Students fork it and submit their coursework as pull requests
against their fork.

The code it inherits does not run.

It is a five-year-unmaintained fork of Microsoft's `MLOpsPython` reference architecture, built
entirely on `azureml.core` — Azure ML SDK v1, which Microsoft has retired in favour of
`azure-ai-ml` and the `az ml` CLI v2. That alone would make it unusable. But a reading of the
source found eight defects, and only some of them are about the SDK:

1. The training script's `main()` is dead code — it reads a CSV with pandas and then accesses the
   result as though it were a scikit-learn `Bunch`. It has never executed.
2. The estimator is constructed with a scikit-learn parameter that was **removed** in version 1.2.
   The root cause is an **unpinned dependency**: nobody changed the code, the library moved.
3. The preprocessing function raises on pandas 2.x, because it computes a mean over a frame that
   still contains a string column.
4. The data-bootstrap path loads the wrong dataset entirely — it fetches sklearn's diabetes data
   while the pipeline then looks for the automobile CSV and raises when it is absent.
5. **The training data is not in the repository at all.** It lived in a workspace that no longer
   exists.
6. The two "A/B test" scoring scripts are identical files. The canary pipeline has always deployed
   a model against itself.
7. The lint-and-unit-test template is commented out of the CI definition. Neither has ever run.
8. The evaluation step cancels its parent pipeline run — an operation with no equivalent in v2.

Beyond the defects, the architecture itself is a liability for teaching. Roughly 600 lines of
Python exist only to construct, publish and then re-trigger a pipeline object — work that v2
expresses as declarative job definitions. Teaching that scaffolding would mean teaching 2020
idioms wearing 2026 names, to a graduate-level cohort, at the cost of the hours the syllabus
actually wants spent on principles.

Additionally, the model artifact carries a latent correctness bug that matters pedagogically:
preprocessing runs *outside* the model and is never persisted, so the caller of the scoring
service is responsible for reproducing imputation statistics it does not have. This is
training/serving skew, sitting in a repository that exists to teach people to avoid it.

## Solution

Rebuild the repository as a **narrow rebuild**, not a port.

Keep the domain code's *shape* — the same story, the same dataset, the same four conceptual steps,
the same file names where students' slides reference them. Delete the orchestration layer
entirely and replace it with declarative `az ml` CLI v2 job definitions. Repair the domain code.
Extract the parts that the course needs to *demonstrate* into small, pure, independently testable
modules.

The rebuilt repository delivers:

- **A four-step training pipeline** — validate → train → evaluate → register — expressed as job
  definitions, submitted with a single blocking command that returns a non-zero exit code on
  failure. The publish-then-trigger-by-REST dance disappears, along with the dead marketplace task
  that performed it.
- **A data contract** as executable code, running as step zero of the pipeline, written against
  the dataset's real defects rather than invented ones.
- **A model artifact that carries its own preprocessing**, logged with an MLflow signature, so the
  scoring interface accepts named columns and the skew is closed.
- **Two deployment paths, taught as a contrast**: a hand-written serving container that students
  build, run locally, and later deploy to Kubernetes; and a no-code managed endpoint that the
  platform builds from the registered model. The comparison between them is a teaching objective,
  not an accident.
- **MLflow used twice**: against a local file-based tracking store with no cloud involved, and
  against the managed backend with one line changed. Demonstrating that portability is the point.
- **Working CI**: linting and unit tests actually enabled.

The work is sequenced so that **the first two stages require no cloud account at all**, which
means implementation can begin before any subscription, tenant or permission question is settled.

## User Stories

**Instructor — preparing the course**

1. As the instructor, I want the repository to run end-to-end on a current Azure ML SDK, so that I can capture screenshots that will still be accurate when I teach.
2. As the instructor, I want the training pipeline submitted by a single blocking command, so that a live demo has one thing to fail rather than four.
3. As the instructor, I want the pipeline to fail loudly and visibly when a step fails, so that failure is demonstrable rather than silent.
4. As the instructor, I want the orchestration expressed declaratively, so that I can show a student the whole pipeline on one slide.
5. As the instructor, I want the first two stages of the rebuild to need no cloud account, so that my build is not blocked on institutional approvals.
6. As the instructor, I want every direct dependency pinned to an exact version, so that this repository does not decay the way its predecessor did.
7. As the instructor, I want the story of *why* its predecessor decayed preserved in the repository history, so that I can teach reproducibility from a real incident rather than a hypothetical.
8. As the instructor, I want the repository to contain no dead, duplicated or never-executed code, so that a student reading it is not learning from something that has never worked.
9. As the instructor, I want a documented acceptance run I can perform in a clean account following only the README, so that I discover hidden manual setup before my students do.
10. As the instructor, I want the serving container finished early, so that the container-and-orchestration half of the course is unblocked independently of any cloud work.
11. As the instructor, I want service-principal creation isolated in one standalone script, so that I can run it on students' behalf if their institution withholds the permission.
12. As the instructor, I want teardown of billable resources documented as the final step of every flow, so that forgotten resources do not accumulate charges.

**Student — learning from the repository**

13. As a student, I want to fork the repository and get a working baseline, so that my first hour is spent learning rather than debugging someone else's rot.
14. As a student, I want to train the model on my own laptop in seconds with no cloud credentials, so that I can experiment freely before touching a managed platform.
15. As a student, I want my local training run to produce a real tracked experiment, so that I understand experiment tracking before I meet a hosted backend.
16. As a student, I want to point the same training code at a managed backend by changing one line, so that I understand the tracking tool is portable and the platform is a backend.
17. As a student, I want the model artifact to include its own preprocessing, so that I can call it without knowing how it was trained.
18. As a student, I want the model's input schema declared and enforced, so that a malformed request fails with a clear message instead of a wrong prediction.
19. As a student, I want to build the serving container myself from a readable Dockerfile, so that I understand what a platform does for me before I let it.
20. As a student, I want the serving container to expose a health endpoint, so that I have something real to point a Kubernetes probe at.
21. As a student, I want the same container image I built to be what my Kubernetes manifests and Helm charts deploy, so that the sections of the course connect.
22. As a student, I want to see the platform build an equivalent image from the model registry, so that I can judge the trade-off between control and convenience.
23. As a student, I want the data validated before training begins, so that I experience a data quality gate stopping a bad run.
24. As a student, I want the data contract written against defects that are genuinely in the data, so that the exercise is not artificial.
25. As a student, I want the training data versioned as a platform asset rather than referenced by path, so that a model's lineage identifies exactly which data produced it.
26. As a student, I want a model-quality threshold that blocks registration when it is not met, so that I understand a metric gate as a form of test.
27. As a student, I want linting and unit tests to run on my pull request, so that I get feedback before a human reviews it.
28. As a student, I want my coursework to be a pull request against my own fork, so that I practise the workflow I will use professionally.
29. As a student, I want the CI/CD definitions and the ML job definitions to sit side by side in one repository, so that the two-artifact problem is visible rather than described.
30. As a student, I want a README that takes me from an empty account to a scored prediction, so that I can recover independently if I fall behind.
31. As a student, I want to deploy the trained model to a managed endpoint and call it from an HTTP client, so that I complete the loop from data to a served prediction.
32. As a student, I want teardown instructions for everything I create, so that I do not exhaust my credits by accident.
33. As a student, I want to run the whole container portion of the course on my own machine, so that I need no cloud quota for it.
34. As a student, I want the repository to work on Apple Silicon as well as x86, so that my laptop is not the reason I fall behind.

**Maintainer — keeping it alive across cohorts**

35. As the maintainer, I want the data-quality rules in one module behind one call, so that changing the dataset does not mean hunting through the training script.
36. As the maintainer, I want the model's preprocessing and estimator constructed in one place, so that training, retraining and serving cannot drift apart.
37. As the maintainer, I want the promote-or-reject policy extracted as a pure function, so that I can change the threshold policy without touching the platform integration.
38. As the maintainer, I want the pure logic tested without any cloud dependency, so that the test suite runs in seconds on any machine.
39. As the maintainer, I want the tests to assert on behaviour rather than on a memorised numeric output, so that they do not break every time the model changes harmlessly.
40. As the maintainer, I want a single lockfile committed, so that a cohort two years from now gets the environment I validated.
41. As the maintainer, I want out-of-scope material deleted rather than left dormant, so that maintenance cost tracks what is actually taught.
42. As the maintainer, I want the declarative definitions schema-validated by the platform CLI, so that a typo fails fast rather than mid-run.

## Implementation Decisions

### Architectural direction

- **Narrow rebuild, not a port.** The domain code's shape is preserved and repaired; the
  orchestration layer is deleted outright. The package that exists solely to build, publish and
  trigger a pipeline object ceases to exist — 21 of 23 Python files are removed, and the
  orchestration collapses into six declarative job definitions.
- **Declarative job definitions are the primary interface.** Compute, environment, data asset,
  pipeline, endpoint and deployment are all YAML. Submission is a single blocking CLI call that
  streams logs and exits non-zero on failure. This replaces the publish → fetch-ID →
  REST-invoke → poll sequence and removes a dependency on a deprecated marketplace CI task.
- **One Python SDK submission script is retained deliberately**, as a short programmatic contrast
  during teaching. Nothing in the repository's critical path depends on it.
- **The evaluation step's parent-cancellation behaviour is replaced by exit codes.** A failing
  quality gate exits non-zero, the job fails, and registration never runs. The semantics differ
  from the original (the run reports as *failed*, not *cancelled*); this is accepted and is the
  better lesson.

### Module design

Five **deep, pure modules** — no I/O, no cloud, no framework coupling — form the testable core:

| Module | Interface | Responsibility |
|---|---|---|
| **data contract** | `validate(frame) → ValidationReport` | Every data-quality rule for the dataset, behind one call |
| **model factory** | `build_pipeline() → Pipeline` | All preprocessing, imputation, scaling and estimator selection in one place |
| **quality gate** | `decide(candidate, incumbent, policy) → GateDecision` | The promote-or-reject policy, as a pure function |
| **metrics** | `get_model_metrics(model, features, target) → dict` | Metric computation |
| **split** | `split_data(frame) → train/test` | Deterministic partitioning |

The **model factory** is the load-bearing extraction. Because training, any future retraining, and
serving all obtain the estimator from it, the preprocessing cannot diverge between them. This is
what closes the training/serving skew described in the problem statement.

The **quality gate** is the second. In the inherited code the promote-or-reject decision is
entangled with platform run-cancellation. Extracted as a pure function it becomes something that
can be unit-tested, demonstrated, and deliberately broken in class.

Thin adapters sit at the I/O boundary and are not unit-tested: the experiment-tracking wrapper,
the four command-line entrypoints (argparse shells over the modules above), and the serving
model loader.

### Model artifact

The registered artifact is a **scikit-learn `Pipeline`**, not a bare estimator:

```
Pipeline([
  ("prep",  ColumnTransformer(...)),   # numeric coercion, imputation, drop the free-text column
  ("scale", StandardScaler()),          # replaces the removed `normalize` parameter
  ("model", LinearRegression()),
])
```

Logged with an **MLflow signature and an input example**, so the model declares a schema of named
columns. Three consequences: no-code deployment produces an endpoint whose contract is human
readable; the serving application needs no duplicated preprocessing; and behavioural tests become
expressible against meaningful inputs.

**The algorithm itself is unchanged.** Linear regression on this dataset trains in milliseconds
and is explainable. The subject is operations, not modelling.

### Data

- The dataset is **committed to the repository as a seed fixture** (~400 rows). Determinism in a
  classroom outweighs purity; an upstream download that fails at the start of a session is not a
  risk worth taking.
- The pipeline **registers it as a versioned platform Data asset**, and every job consumes it *by
  asset version, never by path*. The distinction is taught explicitly: the committed file is a
  seed, the asset is the versioned truth, and the version identifier is what appears in a model's
  lineage.
- Validation is **step zero of the pipeline**, not a side script — a failing contract stops the
  run before compute is spent on training.
- The contract is written against the dataset's **actual defects**, which are already documented
  by the inherited preprocessing code: a numeric column containing sentinel non-numeric values, a
  known row count, and a target that must be positive.

### Deployment — two paths, deliberately

| | Hand-built path | Managed path |
|---|---|---|
| Model source | trained locally, local tracking store | pipeline → model registry |
| Image | hand-written multi-stage Dockerfile, non-root | built by the platform |
| Registry | public registry, then a cloud container registry | cloud container registry, automatic |
| Runtime | local container engine, then Kubernetes | managed online endpoint |
| Teaching point | *"I control the image"* | *"the platform controls it — here is the trade"* |

Both are first-class. The hand-built serving application is a small HTTP service exposing a
prediction endpoint and a health endpoint, loading the model through its tracking flavour. Its
image is the artifact that the Kubernetes manifests and Helm charts consume — which also gives
the previously purposeless canary/blue-green charts something real to deploy.

The managed path is reachable **only because the model carries its own preprocessing and a
signature**; a bare estimator would produce an endpoint with an undocumented positional contract.

### Experiment tracking

The tracking library is used in **both** configurations from identical code:

- With **no tracking URI configured**, it writes to a local file-based store. No cloud, no
  credentials, no account. This is what the container portion of the course uses.
- With the **tracking URI pointed at the managed workspace**, the same code runs against the
  managed backend, and the registry, model versions and no-code deployment become available.

The training code is therefore **backend-agnostic by construction** — it reads the tracking
destination from the environment and never hard-codes a provider. This is a design constraint,
not an afterthought; it also makes local debugging fast, since training can be exercised on a
laptop and only submitted to the platform to prove orchestration.

### Repository, hosting and identity

- The repository is **public**, and students **fork it**. Coursework arrives as pull requests
  against their own fork, reviewable from one account.
- The **CI/CD platform is separate from the code host**: each student runs pipelines in their own
  organisation, connected to their own fork. This is a realistic industrial split and it makes
  branch-versus-pull-request trigger behaviour demonstrable rather than theoretical.
- **Service-principal creation lives in a standalone bootstrap script**, never assumed inside a
  pipeline. Credentials are read from a variable group regardless of who created them, so the
  instructor can run the identical script on students' behalf without any code change.

### Dependencies

- **Python 3.11** — chosen for the widest overlap with the platform's curated base images, the
  cloud SDK, and the tracking library. Boring and universal beats current.
- **Every direct dependency pinned to an exact version, with a committed lockfile.** Given that an
  unpinned dependency is what killed the predecessor, anything else would be indefensible.
- **Three dependency manifests**, down from four, each with one obvious job: the platform training
  environment, the serving container, and the development/CI environment.
- **No `uv` or Poetry.** Both are better tools; both are one more thing to explain in a course
  that already carries a container engine, an orchestrator, a package manager for it, a cloud ML
  platform, a tracking library, a validation library and a CI system.

### Build sequence

| Stage | Cloud needed | Deliverable | Checkpoint |
|---|---|---|---|
| **A** | no | Skeleton, data fixture, the five modules, the four entrypoints, tests, pins | Training runs on a laptop and produces a tracked model with a signature |
| **B** | no | Serving application, Dockerfile, manifest | Container runs; prediction endpoint returns a prediction; health endpoint returns 200 |
| **C** | yes | Data asset, compute, environment, pipeline definitions | Four-step pipeline runs green from a single blocking submit |
| **D** | yes | Endpoint and deployment definitions | Model deployed and scored over HTTP; smoke test passes |
| **E** | yes | CI, CD and pull-request definitions; lint/test template enabled | Green run against a fork |
| **F** | yes | Acceptance run in a clean account, README only | See Testing Decisions |

**Stages A and B require no cloud account**, so implementation begins immediately and is not
gated on institutional approvals. Stage B additionally completes the artifact that the container
and orchestration sections of the course depend on.

### Deletions

Removed as out-of-syllabus or superseded: the R-and-Databricks training path; the parallel
batch-scoring path; the three near-duplicate scoring scripts; the v1 inference and deployment
configuration files; the platform image-build helper; the entire orchestration package; the
data-bootstrap helper; the project-renaming helper; the explainability integration; and one of the
two infrastructure-provisioning definitions.

Retained and rewritten: training, evaluation and registration entrypoints; the unit test; the
scoring smoke test; the Helm charts and load-generation script; the Helm install/upgrade
templates; the pull-request and code-quality CI templates; the exploratory notebook.

## Testing Decisions

### What makes a good test here

A good test in this repository asserts on **external behaviour through a module's public
interface**, and would survive a reasonable internal refactor. It does not reach into private
helpers, does not assert on call ordering, and does not encode a memorised output value.

The inherited test suite is the anti-pattern, and it is instructive: its single test asserts that
a metric equals `0.029843893480257067`. That assertion is a fingerprint of one particular
implementation. It tells you nothing about correctness, it breaks on any harmless change, and —
because the CI template that would have run it was commented out — it has never once executed.
The replacement asserts on properties: that the expected metric keys are present, that a perfect
prediction yields zero error, that error increases as predictions worsen.

### What gets tested in this workstream

**Unit tests, test-first, for the five deep pure modules only.** These run in seconds, need no
credentials, no network and no container engine.

- **data contract** — a conforming frame passes; each rule fails independently when violated;
  sentinel non-numeric values are detected; the report identifies which rule failed and where.
- **model factory** — returns an unfitted pipeline with the expected stages; fits on a frame
  containing the dataset's real defects without pre-cleaning; the fitted object predicts from raw
  named columns; preprocessing statistics survive a serialise/deserialise round trip.
- **quality gate** — promotes when the candidate improves on the incumbent; rejects when it does
  not; handles the no-incumbent case; respects the configured threshold; is a pure function of its
  arguments.
- **metrics** — expected keys present; a perfect predictor scores zero error; error is monotonic
  in prediction quality.
- **split** — partitions are disjoint and exhaustive; the ratio is honoured; the same seed
  produces the same partition.

**Enabling the CI template is itself a deliverable.** Linting and unit tests run on every pull
request — for the instructor, and for every student's coursework.

### Acceptance testing

Stage F is a manual acceptance run, and it is the gate on slide production. Performed in a
**clean account**, following **only the README**, as a student would: submit the pipeline, watch
validate → train → evaluate → register succeed, confirm the run and its metrics are visible in
tracking, confirm the model is registered against a data asset version, deploy to a managed
endpoint, score it over HTTP, and **delete the endpoint**.

"Clean account" is the whole point of the exercise — it is what catches undocumented manual setup
that has quietly accumulated in the developer's own environment.

Stage B has its own earlier checkpoint, verified manually and needing no cloud account: build the
image, run it, call the prediction endpoint, call the health endpoint, on **both x86 and Apple
Silicon**.

### Prior art

There is little worth inheriting. The single existing unit test is the anti-pattern described
above. The existing scoring smoke test is sound in shape — plain HTTP against a deployed service —
and is retained, retargeted at the new endpoint. Its role is integration verification during Stage
D, not part of the unit suite.

## Out of Scope

**Deferred to later course-content work, though this repository will eventually host them:**

- **Kubernetes manifests** for the orchestration section. The serving image they deploy is a
  Stage B deliverable; the manifests themselves are content work.
- **Drift detection.** The tooling choice is settled but building it belongs to the
  testing-and-observability section.
- **Metrics instrumentation, scraping and dashboards** for the observability section.
- **Behavioural model tests** (domain-property assertions about prediction direction). Good
  material, and the signature work in this PRD is what makes them expressible — but they are model
  quality, not the rebuild.
- **Integration tests for the serving application** and **smoke tests for the tracking adapter**.
  Both were considered and explicitly deferred; Stage B's manual checkpoint and Stage D's smoke
  test cover them for now.

**Not in this repository at all:**

- Slide production, translation, and any course material.
- Assessment design.
- The container-fundamentals lab material.
- Provisioning of shared cluster infrastructure.
- Infrastructure-as-code teaching. One provisioning definition is retained as an opaque
  convenience; the subject is owned by a different module.

**Permanently removed:** R and Databricks training; parallel batch scoring; model explainability.

## Further Notes

**The reproducibility incident is an asset — preserve it.** This repository's predecessor died
because one dependency was unpinned: nobody changed the code, and a library removed a parameter
underneath it. A real repository, training a real model, killed by a missing version pin. That is
worth more as teaching material than any invented example, and it is the reason the pinning
decision above is non-negotiable rather than stylistic.

**Two of the eight inherited defects are pedagogically useful and should be preserved in history
rather than quietly fixed** — the unpinned dependency, and the CI template that was commented out
so the tests never ran. Both are seeded-defect material.

**Estimate: 5–7 days** of focused engineering, of which Stages A and B are roughly two. The
original estimate was 3–5 days; it assumed migrating working code.

**Sequencing risk.** Stages C through F depend on a cloud account with permission to create an
application registration and assign it a subscription role — two distinct grants, in two distinct
systems, typically owned by two different teams. Neither has been confirmed for the student cohort
at the time of writing. The build sequence above is deliberately ordered so that this uncertainty
delays nothing for the first two stages, and the bootstrap script is deliberately standalone so
that a refusal costs a change of operator rather than a change of architecture.

**The course build plan is the upstream document** for anything not covered here — section-by-
section hour budgets, reuse analysis, assessment design, and institutional logistics. It is
maintained privately and is not part of this repository.
