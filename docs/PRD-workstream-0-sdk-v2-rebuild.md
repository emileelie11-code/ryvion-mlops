# PRD — Workstream 0: the Ryvion MLOps teaching repo

**Status:** in progress — the local core is built and merged
**Scope:** this repository only
**Blocks:** slide production for the course sections on CI/CD, MLOps pipelines, and testing/observability

---

## Revision 2 — the cloud dependency is withdrawn (21 Aug 2026)

This PRD originally specified a rebuild onto Azure ML SDK v2 and the `az ml` CLI. **That premise
has been withdrawn.** Re-reading the course syllabus directly — rather than a summary of it —
settled the question: across all three pages it contains **zero** occurrences of Azure, AWS, GCP
or "managed", and two of "cloud", both of which are context rather than content (the
programme block's name, and a *prerequisite*).

Every tool the syllabus names by name — GitHub Actions, Docker, Kubernetes, Helm, Prometheus,
Grafana, MLflow — runs on a laptop. Two of them are named with no alternative offered, and both
were the ones a cloud platform made *harder* to satisfy rather than easier.

**What this changes:** the orchestration target, the deployment target, and the access model.
**What it does not change:** the module design, the model artifact, the data contract, the
testing decisions, or anything already built. The engineering core of this document stands as
written; the sections below are marked where they were revised.

**What it removes:** per-student cloud subscriptions and their credit budget, two directory
permissions nobody had confirmed, a CI platform organisation per student with a multi-week
approval lead time, a shared managed Kubernetes cluster with per-student namespace provisioning,
and a continuously-billing endpoint that had to be remembered and torn down.

**Progress at revision time:** six slices merged, 156 tests green, verified end to end on a
clean machine — the toolchain skeleton, local training producing a signed model artifact, the
data contract, the quality gate, and the hand-built serving container.

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
entirely and replace it with declarative CI workflows. Repair the domain code.
Extract the parts that the course needs to *demonstrate* into small, pure, independently testable
modules.

The rebuilt repository delivers:

- **A four-step training pipeline** — validate → train → evaluate → register — each step an
  ordinary command-line entrypoint that exits non-zero on failure, chained as dependent CI jobs.
  The publish-then-trigger-by-REST dance disappears, along with the dead marketplace task that
  performed it. Every step remains runnable by hand, which is what makes the pipeline teachable.
- **A data contract** as executable code, running as step zero of the pipeline, written against
  the dataset's real defects rather than invented ones.
- **A model artifact that carries its own preprocessing**, logged with an MLflow signature, so the
  scoring interface accepts named columns and the skew is closed.
- **Two deployment paths, taught as a contrast** *(revised)*: a hand-written serving container
  that students build, run locally, and deploy to Kubernetes themselves — and a managed endpoint
  that a platform builds from the registered model, **demonstrated once by the instructor**
  rather than operated by each student. The comparison remains a teaching objective; only which
  side is hands-on has flipped, and the hands-on side is now the one that teaches more.
- **MLflow used twice** *(revised)*: against a local file-based store, then against a
  database-backed store where the model registry lives. The portability point still lands, and
  lands more honestly — the tracking destination is configuration that never appears in the
  code, so the same training script runs against a file store, a local database, or a remote
  tracking server without modification.
- **Working CI**: linting and unit tests actually enabled.

**No stage requires a cloud account.** The original sequencing existed to get useful work done
before an institutional approval landed; the approval is no longer needed at all.

## User Stories

**Instructor — preparing the course**

1. As the instructor, I want the repository to run end-to-end on a current, maintained toolchain, so that I can capture screenshots that will still be accurate when I teach.
2. As the instructor, I want the training pipeline submitted by a single blocking command, so that a live demo has one thing to fail rather than four.
3. As the instructor, I want the pipeline to fail loudly and visibly when a step fails, so that failure is demonstrable rather than silent.
4. As the instructor, I want the orchestration expressed declaratively, so that I can show a student the whole pipeline on one slide.
5. As the instructor, I want no stage of the rebuild to need a cloud account, so that neither my build nor my students' work is blocked on an institutional approval.
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
- **Declarative CI workflows are the primary interface** *(revised — was cloud job YAML).* The
  four pipeline steps are dependent jobs in a single workflow on the code host's own CI. Each
  step is an ordinary command-line entrypoint that exits non-zero on failure, so the pipeline
  stops without any platform-specific control flow. This removes the publish → fetch-ID →
  REST-invoke → poll sequence *and* the second CI platform the original plan required.
- **The steps stay runnable by hand.** Every entrypoint works identically on a laptop and in CI.
  That is what makes the pipeline teachable: a student runs a step, sees its exit code, and only
  then sees the workflow that chains them.
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

### Deployment — two paths, deliberately *(revised)*

| | Hand-built path — **students** | Managed path — **instructor demo** |
|---|---|---|
| Model source | trained locally, local tracking store | the model registry |
| Image | hand-written multi-stage Dockerfile, non-root | built by the platform |
| Registry | local registry, then the code host's registry | the platform's registry, automatic |
| Runtime | local container engine, then a local Kubernetes cluster | a managed endpoint |
| Who operates it | every student, throughout | shown once, from the instructor's own account |
| Teaching point | *"I control the image"* | *"the platform controls it — here is the trade"* |

The hand-built path also gains the deployment-strategy work: canary, blue-green and **shadow**
are all expressible with the Helm charts already in this repository, on a local cluster. A
managed endpoint offers traffic splitting but not a genuine shadow deployment — so the local
path is not a downgrade here, it is the only one that covers the full requirement.

Both are first-class. The hand-built serving application is a small HTTP service exposing a
prediction endpoint and a health endpoint, loading the model through its tracking flavour. Its
image is the artifact that the Kubernetes manifests and Helm charts consume — which also gives
the previously purposeless canary/blue-green charts something real to deploy.

Both paths depend on the model carrying **its own preprocessing and a declared signature**; a
bare estimator would produce a service with an undocumented positional contract either way.

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
- **CI runs on the code host** *(revised — was a separate CI organisation per student).* It runs
  on a fork with no setup, no service connection, no organisation to create and no parallelism
  grant to wait on, so a student gets feedback on their first pull request on day one. The
  original split was defensible as industrial realism, but it cost every student an account and
  an approval before a single test could run.
- **No credentials are required at any point** *(revised).* The bootstrap script that created a
  service identity, and the two directory permissions it depended on, are both withdrawn.

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

*(Revised. Stages C–F previously targeted a cloud platform and were blocked on an account that
did not exist. They now target the same local toolchain as A and B, and nothing is blocked.)*

| Stage | Deliverable | Checkpoint | Status |
|---|---|---|---|
| **A** | Skeleton, data fixture, the five modules, the four entrypoints, tests, pins | Training runs on a laptop and produces a tracked model with a signature | **done** |
| **B** | Serving application, Dockerfile, manifest | Container runs; prediction endpoint returns a prediction; health endpoint returns 200 | **done** |
| **C** | The four steps as dependent CI jobs in one workflow | Pipeline runs green on a pull request; a failing contract or a rejected model stops it | to do |
| **D** | Kubernetes manifests — Deployment, Service, probes, resource limits, autoscaling — plus canary / blue-green / shadow on the existing Helm charts | The serving image runs on a local cluster; a probe failure keeps traffic away; load drives a scale-up | to do |
| **E** | Metrics instrumentation in the serving container; a monitoring stack on the local cluster; one dashboard | Predictions and latency visible on a dashboard scraped from the running container | to do |
| **F** | Acceptance run on a clean machine, README only | See Testing Decisions | to do |

**Stage B was also the course's container deliverable** — finishing it unblocked the container
and orchestration sections outright. Stage E is the memory ceiling of the whole course and
should be rehearsed on a typical student laptop rather than a workstation.

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

**Sequencing risk — resolved, and worth recording how.** Stages C through F previously depended
on a cloud account plus two directory permissions, in two systems, owned by two different teams,
none of which had been confirmed. That risk was mitigated by ordering the work so the
uncertainty blocked as little as possible — and then eliminated outright by removing the
dependency. The mitigation was sound; the better move was noticing the dependency was never
required.

**The risk that replaces it is student machines.** The delivery risk is swapped rather than
removed: a container engine on employer-managed laptops now carries the multi-week lead time,
and available memory becomes the ceiling on the observability work. That belongs to the course
plan rather than to this repository, but it is the reason nobody should read this revision as
risk-free.

**The course build plan is the upstream document** for anything not covered here — section-by-
section hour budgets, reuse analysis, assessment design, and institutional logistics. It is
maintained privately and is not part of this repository.
