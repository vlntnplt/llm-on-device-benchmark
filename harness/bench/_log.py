"""Progress, warnings, and fatal errors — all to stderr.

stdout is reserved for machine-readable output (the plan table, results paths),
so everything human goes here and the tool composes cleanly in a pipeline.
"""

from __future__ import annotations

import sys
from typing import NoReturn


def log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def warn(msg: str) -> None:
    print(f"⚠️  {msg}", file=sys.stderr, flush=True)


def die(msg: str) -> NoReturn:
    raise SystemExit(f"✗ {msg}")
