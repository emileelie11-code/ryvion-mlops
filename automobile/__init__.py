"""Domain code for the `ryvion-mlops` teaching repository.

The package is deliberately shallow at the top level. Two kinds of module live
underneath it:

``automobile.entrypoints``
    Thin argparse shells, one per pipeline step (validate, train, evaluate,
    register). They parse arguments and delegate; they hold no domain logic and
    are not unit tested.

The pure domain modules (data contract, model factory, quality gate, metrics,
split) are added by the slices that need them and are where the unit tests
point.
"""

__version__ = "0.1.0"
