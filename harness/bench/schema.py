"""The two contract boundaries, as validators.

`events.schema.json` is what a backend exe emits — we check it on the way *in*,
before any metric trusts a single stamp. `results.schema.json` is what the harness
writes — we check it on the way *out*, before anything lands in results/. A
backend that emits garbage fails loudly here, at the seam, not three functions
deep in a median.
"""

from __future__ import annotations

import json
from functools import cache

from jsonschema import Draft202012Validator

from .config import REPO

SCHEMA_DIR = REPO / "schema"  # the shared contract, at the repo root


class SchemaError(ValueError):
    """An object failed to validate against its contract schema."""


@cache
def _validator(filename: str) -> Draft202012Validator:
    schema = json.loads((SCHEMA_DIR / filename).read_text())
    Draft202012Validator.check_schema(schema)  # the schemas themselves must be valid
    return Draft202012Validator(schema)


def _check(obj: object, filename: str, label: str) -> None:
    validator = _validator(filename)
    errors = sorted(validator.iter_errors(obj), key=lambda e: list(e.absolute_path))
    if errors:
        lines = "\n".join(
            f"  • {'/'.join(map(str, e.absolute_path)) or '<root>'}: {e.message}"
            for e in errors[:8]
        )
        more = "" if len(errors) <= 8 else f"\n  … and {len(errors) - 8} more"
        raise SchemaError(f"{label} failed {filename}:\n{lines}{more}")


def validate_events(obj: object, *, label: str = "events") -> None:
    """Raise SchemaError unless `obj` is a valid backend events object (input)."""
    _check(obj, "events.schema.json", label)


def validate_results(obj: object, *, label: str = "results") -> None:
    """Raise SchemaError unless `obj` is a valid aggregated results object (output)."""
    _check(obj, "results.schema.json", label)
