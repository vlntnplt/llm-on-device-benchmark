"""Load on-device benchmark results into tidy DataFrames for cross-machine
comparison.

Each `bench run` writes one `<backend>-results.json` (validated against
`results.schema.json`). Run the benchmark on N machines and you get N×
sets of those files. The loaders fan over a directory tree, flatten every
file, and concatenate — machine identity rides *inside* each file (the `machine`
block), so merging is just "add machine columns and stack rows".

Recommended layout — one subdir per machine so filenames don't collide:

    results/
      3090-box/   ggml-results.json
      m1-max/     ggml-results.json

Flat files directly under the root also load; their machine label is then derived
from the in-file `machine` block instead of the directory name.

Three frames, one per kind of measurement:

- `load_results` — the validation job: one row per (machine, backend, model,
  quant, provider). Every `[p50, max]` stat explodes into `<name>_p50` /
  `<name>_max` (a null stat — e.g. VRAM on a CPU EP — explodes to NaN/NaN,
  never 0). Geometry scalars ride along (`geo_*`). Cells that produced no
  timing still get a row, flagged by `status`:

      ok         a scored job with metrics
      too_slow   backstop killed, or below the usable tok/s floor
      errored    attempted but produced no sample — crash/OOM, not slowness
      unhealthy  the (model, provider) failed its brain-check; nothing ran

- `load_memory` — the memory cost curve: one row per allocator context point
  from the sweep's `geometry.memory_points`, with `n_ctx` and the pooled
  `weights_mb` / `kv_mb` / `compute_mb`. Exact allocator numbers (no spread);
  what an estimator fits, and what the report reads at the job's context.

- `load_sweeps` — one row per sweep point: `kind` ("prefill" | "decode").
  Prefill rows are the instrumented pass's chunks (`chunk_ms` marginal,
  `ttft_ms` cumulative at depth `tokens`); decode rows the tps spread per
  `kv_fill`. Points survive a non-ok sweep status — partial data still
  informs.

- `load_probes` — one row per ceiling point: `kind` ("gemm" | "h2d" | "d2h" |
  "d2d") with the measured `tflops` or `gbs`.

A missing number is *visible*, not silently absent.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd

# The results schema this loader understands; in lockstep with
# results.schema.json's `schema_version`. A mismatch is a loud error, not a
# silently-misaligned frame.
SCHEMA_VERSION = "2"

# Scalar fields copied straight off each run.
_RUN_KEYS = ("provider", "device", "model", "quant", "healthy", "vram_method")

# Geometry scalars worth having as columns (the full block incl. per-layer
# typing stays in the JSON for consumers that need it).
_GEO_KEYS = ("n_layer", "n_params", "file_bytes", "n_ctx_train")


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


def _docs(root: str | Path):
    """(label, machine-base columns, doc) per results file, version-checked."""
    root = Path(root)
    for f in sorted(root.glob("**/*-results.json")):
        doc = json.loads(f.read_text())
        if doc.get("schema_version") != SCHEMA_VERSION:
            raise ValueError(
                f"{f}: results schema_version={doc.get('schema_version')!r}, "
                f"loader expects {SCHEMA_VERSION!r}"
            )
        machine = doc["machine"]
        memory = machine.get("memory") or {}
        base = {
            "machine": _machine_label(f, root, machine),
            "os": machine["os"],
            "cpu": machine["cpu"],
            "gpu": ", ".join(machine.get("gpus") or []) or "cpu",
            "ram_gb": memory.get("total_gb"),
            "ram_channels": memory.get("channels"),
            "ram_mts": memory.get("configured_mts"),
            "backend": doc["backend"],
        }
        yield base, doc


def _explode(stats: dict, into: dict) -> None:
    """Split each `[p50, max]` (or null) stat into `<name>_p50` / `<name>_max`."""
    for name, stat in stats.items():
        p50, mx = stat if stat else (None, None)
        into[f"{name}_p50"] = p50
        into[f"{name}_max"] = mx


def load_results(root: str | Path = "results") -> pd.DataFrame:
    """The validation-job frame: one row per (machine, backend, model, quant,
    provider). Raises ValueError on a schema_version mismatch; empty DataFrame
    when no results files are found."""
    rows: list[dict] = []
    for base, doc in _docs(root):
        for run in doc["runs"]:
            row = {**base, **{k: run[k] for k in _RUN_KEYS}}
            row["unhealthy_reason"] = run.get("unhealthy_reason")
            for key in _GEO_KEYS:
                row[f"geo_{key}"] = (run.get("geometry") or {}).get(key)
            job = run["job"]
            row["task"] = job["task"] if run["healthy"] else None
            row["status"] = job["status"] if run["healthy"] else "unhealthy"
            row["sweep_status"] = run["sweep"]["status"]
            if job["status"] == "ok":
                _explode(job["metrics"], row)
                _explode(job["memory"], row)
                row["sample_completion"] = (job["sample_completions"] or [None])[0]
            rows.append(row)
    return pd.DataFrame(rows)


def load_sweeps(root: str | Path = "results") -> pd.DataFrame:
    """The sweep-point frame: one row per measured point, `kind` prefill/decode.

    Prefill rows are the instrumented pass's chunks: `tokens` is the prompt
    depth the chunk reached (context + chunk), `chunk_ms` the chunk's own cost
    (the marginal curve — its slope is the attention term), `ttft_ms` the
    cumulative wall time through that depth. Decode rows carry the tps spread
    at each kv_fill."""
    rows: list[dict] = []
    for base, doc in _docs(root):
        for run in doc["runs"]:
            run_base = {**base, **{k: run[k] for k in _RUN_KEYS},
                        "sweep_status": run["sweep"]["status"]}
            cum = 0.0
            for p in run["sweep"]["prefill"]:
                cum += p["ms"]
                rows.append({**run_base, "kind": "prefill",
                             "tokens": p["context"] + p["tokens"], "kv_fill": None,
                             "chunk_ms": p["ms"], "ttft_ms": round(cum, 2)})
            for p in run["sweep"]["decode"]:
                rows.append({**run_base, "kind": "decode", "tokens": p["tokens"],
                             "kv_fill": p["kv_fill"], "tps_p50": p["tps_p50"],
                             "tps_min": p["tps_min"], "tps_max": p["tps_max"],
                             "n_reps": p["n_reps"]})
    return pd.DataFrame(rows)


def load_memory(root: str | Path = "results") -> pd.DataFrame:
    """The memory-cost-curve frame: one row per allocator context point."""
    rows: list[dict] = []
    for base, doc in _docs(root):
        for run in doc["runs"]:
            for p in (run.get("geometry") or {}).get("memory_points") or []:
                b = p["buffers"]
                rows.append({**base, **{k: run[k] for k in _RUN_KEYS},
                             "n_ctx": p["n_ctx"],
                             "weights_mb": round(sum(x["model_bytes"] for x in b) / 1e6, 1),
                             "kv_mb": round(sum(x["context_bytes"] for x in b) / 1e6, 1),
                             "compute_mb": round(sum(x["compute_bytes"] for x in b) / 1e6, 1)})
    return pd.DataFrame(rows)


def load_probes(root: str | Path = "results") -> pd.DataFrame:
    """The device-ceiling frame: one row per probe point."""
    rows: list[dict] = []
    for base, doc in _docs(root):
        for probe in doc.get("probes") or []:
            probe_base = {**base, "provider": probe["provider"], "device": probe["device"],
                          "status": probe["status"]}
            for g in probe["gemm"]:
                rows.append({**probe_base, "kind": "gemm", "m": g["m"], "n": g["n"],
                             "k": g["k"], "dtype": g["dtype"], "tflops": g["tflops_p50"],
                             "gbs": None, "n_reps": g["n_reps"]})
            for c in probe["copy"]:
                rows.append({**probe_base, "kind": c["kind"], "m": None, "n": None,
                             "k": None, "dtype": None, "tflops": None,
                             "gbs": c["gbs_p50"], "n_reps": c["n_reps"]})
    return pd.DataFrame(rows)
