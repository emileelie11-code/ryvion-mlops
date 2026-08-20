"""The local half of this repository must stay cloud-free.

Stage A of the rebuild is the part a student runs on a laptop with no account,
no credentials and no network. That promise is only credible if it is enforced:
a single ``import azure.ai.ml`` somewhere in the domain package would turn "runs
anywhere" into "runs once someone has a subscription", and nobody would notice
until a student did.

The check is a source scan rather than an import-time one, so it fails on a
machine where no cloud SDK is installed at all - which is exactly the machine it
is protecting.
"""

import ast
from pathlib import Path

import pytest

PACKAGE = Path(__file__).resolve().parents[1] / "automobile"
TESTS = Path(__file__).resolve().parent

# Top-level distributions that mean "this code now needs a cloud account".
CLOUD_SDK_ROOTS = frozenset(
    {
        "azure",
        "azureml",
        "azure_ai_ml",
        "azureml_mlflow",
        "adlfs",
        "msrest",
        "msrestazure",
        "boto3",
        "botocore",
        "sagemaker",
        "google",
        "googleapiclient",
        "databricks",
    }
)


def imported_roots(source: Path) -> set[str]:
    """The top-level module name of every import in one file."""
    roots: set[str] = set()
    for node in ast.walk(ast.parse(source.read_text(encoding="utf-8"), str(source))):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            roots.add(node.module.split(".")[0])
    return roots


def python_files(root: Path) -> list[Path]:
    return sorted(path for path in root.rglob("*.py") if "__pycache__" not in path.parts)


@pytest.mark.parametrize("source", python_files(PACKAGE), ids=lambda p: p.name)
def test_no_module_in_the_domain_package_imports_a_cloud_sdk(source):
    offending = imported_roots(source) & CLOUD_SDK_ROOTS

    assert not offending, f"{source.name} imports a cloud SDK: {sorted(offending)}"


@pytest.mark.parametrize("source", python_files(TESTS), ids=lambda p: p.name)
def test_no_test_imports_a_cloud_sdk(source):
    offending = imported_roots(source) & CLOUD_SDK_ROOTS

    assert not offending, f"{source.name} imports a cloud SDK: {sorted(offending)}"


def test_the_scan_actually_reads_imports():
    """Guards the guard: a scan that finds nothing would pass vacuously."""
    roots = set().union(*(imported_roots(source) for source in python_files(PACKAGE)))

    assert {"mlflow", "pandas", "sklearn"} <= roots
