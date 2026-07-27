"""`bench merge` — extend an existing run's raw trace with a newer one.

The amend path: a submission grows a model (or repairs a cell) without
re-measuring everything. Both traces must come from the *same experiment* —
same backend, same machine, same stack versions, same job shape — the merge
refuses anything else, loudly, so a submission never silently mixes
measurement conditions. New cells land next to the old ones; a cell measured
in both traces is replaced by the new measurement (logged). Ceiling probes are
machine properties, so the base's are kept.

Output is a fresh raw + re-aggregated results in `--out`, ready for
`bench publish`.
"""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path

from .. import aggregate, schema
from .._log import log
from .aggregate import RAW_SCHEMA_VERSION, read_raw

# Fields that identify the box; a mismatch means "not the same machine".
MACHINE_IDENTITY = ("host", "os", "cpu", "cpu_cores", "cpu_threads", "gpus")


def _cell_key(cell: dict) -> tuple[str, str, str]:
    return (cell["model"], cell["quant"], cell["provider"])


def _versions(raw: dict) -> dict:
    """The stack `versions` block, from the first events object in the trace."""
    traces = [p.get("trace") for p in raw.get("probes", [])]
    for cell in raw.get("cells", []):
        traces += cell.get("gate_spawns", [])
        traces.append((cell.get("sweep") or {}).get("trace"))
        traces += (cell.get("job") or {}).get("spawns", [])
    for t in traces:
        if t and t.get("events"):
            return t["events"]["versions"]
    raise SystemExit("raw trace holds no events at all — nothing to merge")


def _check(name: str, base, new) -> None:
    if base != new:
        raise SystemExit(f"refusing to merge: {name} differs\n  base: {base!r}\n  new:  {new!r}")


def merge_raw(base: dict, new: dict) -> dict:
    """The pure merge: `new`'s cells into `base`, after proving both traces
    measured the same experiment."""
    for raw, label in ((base, "base"), (new, "new")):
        if raw.get("schema_version") != RAW_SCHEMA_VERSION:
            raise SystemExit(
                f"{label}: raw schema_version={raw.get('schema_version')!r}, "
                f"harness expects {RAW_SCHEMA_VERSION!r}"
            )
    _check("backend", base["backend"], new["backend"])
    for field in MACHINE_IDENTITY:
        _check(f"machine.{field}", base["machine"].get(field), new["machine"].get(field))
    if base["machine"].get("memory") != new["machine"].get("memory"):
        log("  machine.memory differs (e.g. one run without sudo) — keeping base's")
    _check("sampling", base["sampling"], new["sampling"])
    _check("job_spawns", base["job_spawns"], new["job_spawns"])
    _check("job_iters", base["job_iters"], new["job_iters"])
    _check("versions", _versions(base), _versions(new))

    merged = dict(base)
    cells = {_cell_key(c): c for c in base["cells"]}
    for cell in new["cells"]:
        key = _cell_key(cell)
        log(f"  {'replacing' if key in cells else 'adding'} cell {' '.join(key)}")
        cells[key] = cell
    merged["cells"] = list(cells.values())

    have = {p["provider"] for p in base["probes"]}
    for probe in new["probes"]:
        if probe["provider"] in have:
            log(f"  probe {probe['provider']}: base already has one — keeping base's")
        else:
            merged["probes"] = merged["probes"] + [probe]
    return merged


def _resolve_raw(path: Path) -> Path:
    """A raw file, or a dir holding exactly one `*-raw.json[.gz]`."""
    if path.is_file():
        return path
    hits = sorted(path.glob("*-raw.json.gz")) + sorted(path.glob("*-raw.json"))
    if len(hits) != 1:
        raise SystemExit(f"{path}: expected one raw trace, found {[h.name for h in hits]}")
    return hits[0]


def cmd_merge(args: argparse.Namespace) -> None:
    base = read_raw(_resolve_raw(args.base))
    new = read_raw(_resolve_raw(args.new))
    merged = merge_raw(base, new)

    args.out.mkdir(parents=True, exist_ok=True)
    raw_path = args.out / f"{merged['backend']}-raw.json.gz"
    with gzip.open(raw_path, "wt", encoding="utf-8") as fh:
        json.dump(merged, fh)
    log(f"wrote {raw_path}  ({len(merged['cells'])} cells)")

    results = aggregate.build(merged)
    schema.validate_results(results)
    out_path = args.out / f"{merged['backend']}-results.json"
    out_path.write_text(json.dumps(results, indent=2))
    log(f"wrote {out_path}  ({len(results['runs'])} runs)")
