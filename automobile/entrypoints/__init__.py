"""Command-line entrypoints for the four pipeline steps.

Each module here is an argparse shell over the pure domain modules. The four
steps run in this order and each is submitted as its own job by the pipeline
definition:

1. ``validate``  - stop the run before compute is spent on bad data
2. ``train``     - fit the model pipeline and log it with a signature
3. ``evaluate``  - compare the candidate against the incumbent
4. ``register``  - promote the candidate when the quality gate allows it

The console scripts declared in ``pyproject.toml`` point at the ``main``
function of each module, so ``automobile-train`` and
``python -m automobile.entrypoints.train`` are the same program.
"""

STEPS = ("validate", "train", "evaluate", "register")
