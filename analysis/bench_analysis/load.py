"""Load on-device benchmark results into one tidy DataFrame for cross-machine
comparison.

Each `bench run` writes one `<backend>-results.json` (validated against
`results.schema.json`). Run the benchmark on N machines and you get N×
sets of those files. `load_results` fans over a directory tree, flattens every
file, and concatenates — machine identity rides *inside* each file (the `machine`
block), so merging is just "add machine columns and stack rows".

Recommended layout — one subdir per machine so filenames don't collide:

    results/
      3090-box/   ggml-results.json  tjs-results.json
      m1-max/     ggml-results.json  tjs-results.json

Flat files directly under the root also load; their machine label is then derived
from the in-file `machine` block instead of the directory name.

The frame is long/tidy: one row per (machine, backend, model, quant, provider,
device, task). Every `[p50, max]` stat is exploded into `<name>_p50` / `<name>_max`
columns (a null stat — e.g. VRAM on a CPU EP — explodes to NaN/NaN, never 0).
Cells that produced no timing still get a row, flagged by `status`:

    ok         a timed task with metrics
    too_slow   too slow to score: backstop timeout or below the floor (`timed_out_tasks`)
    errored    attempted but produced no sample — crash/OOM, not slowness (`errored_tasks`)
    unhealthy  the (model, provider) failed its brain-check; no tasks were run

so a missing number is *visible*, not silently absent.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd

# The results schema this loader understands; in lockstep with
# results.schema.json's `schema_version`. A mismatch is a loud error, not a
# silently-misaligned frame.
SCHEMA_VERSION = "1"

# Scalar fields copied straight off each run.
_RUN_KEYS = ("provider", "device", "model", "quant", "healthy", "vram_method")


def _slug(machine: dict) -> str:
    """A filesystem-safe label for a machine with no directory name to borrow — its
    `host` name, else its first GPU, else its CPU string."""
    basis = (
        machine.get("host") or (machine.get("gpus") or [None])[0] or machine.get("cpu", "unknown")
    )
    return re.sub(r"[^a-z0-9]+", "-", basis.lower()).strip("-") or "unknown"


def _machine_label(path: Path, root: Path, machine: dict) -> str:
    """Subdir name when the file lives in `results/<machine>/…` (lets you relabel
    without re-running); otherwise the machine's own `host`, slugged."""
    parent = path.parent
    if parent.resolve() != root.resolve():
        return parent.name
    return _slug(machine)


def _explode(stats: dict, into: dict) -> None:
    """Split each `[p50, max]` (or null) stat into `<name>_p50` / `<name>_max`."""
    for name, stat in stats.items():
        p50, mx = stat if stat else (None, None)
        into[f"{name}_p50"] = p50
        into[f"{name}_max"] = mx


def load_results(root: str | Path = "results") -> pd.DataFrame:
    """Load every `*-results.json` under `root` into one tidy frame.

    Raises ValueError on a schema_version mismatch. Returns an empty DataFrame if
    no results files are found.
    """
    root = Path(root)
    rows: list[dict] = []
    for f in sorted(root.glob("**/*-results.json")):
        doc = json.loads(f.read_text())
        if doc.get("schema_version") != SCHEMA_VERSION:
            raise ValueError(
                f"{f}: results schema_version={doc.get('schema_version')!r}, "
                f"loader expects {SCHEMA_VERSION!r}"
            )
        machine = doc["machine"]
        base = {
            "machine": _machine_label(f, root, machine),
            "os": machine["os"],
            "cpu": machine["cpu"],
            "gpu": ", ".join(machine.get("gpus") or []) or "cpu",
            "backend": doc["backend"],
            "iters": doc["iters"],  # sample-size provenance: timing n = iters×spawns
            "spawns": doc["spawns"],
        }
        for run in doc["runs"]:
            run_base = {**base, **{k: run[k] for k in _RUN_KEYS}}
            run_base["unhealthy_reason"] = run.get("unhealthy_reason")

            if not run["healthy"]:
                rows.append({**run_base, "task": None, "status": "unhealthy"})
                continue

            for t in run["tasks"]:
                row = {**run_base, "task": t["task"], "status": "ok"}
                _explode(t["metrics"], row)
                _explode(t["memory"], row)
                row["sample_completion"] = (t["sample_completions"] or [None])[0]
                rows.append(row)
            for task in sorted(run.get("timed_out_tasks") or []):
                rows.append({**run_base, "task": task, "status": "too_slow"})
            for task in sorted(run.get("errored_tasks") or []):
                rows.append({**run_base, "task": task, "status": "errored"})

    return pd.DataFrame(rows)
