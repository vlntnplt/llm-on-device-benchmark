"""`bench aggregate` — re-derive results from persisted raw traces.

Inference is expensive; aggregation is cheap and changes often. `bench run`
persists the raw per-spawn traces first, then derives results from them — this
command re-runs only that second step, so a metric change is a re-aggregate, not a
re-run. Same `aggregate.build`, same output, byte-for-byte.

`RAW_SCHEMA_VERSION` versions the trace artifact. It is harness-internal — *not* a
backend contract (only events/results are) — so it versions independently of
the results schema. Samples carry raw measurements (deduped per-PID VRAM, GTT
folded into RSS), so every derivation stays a re-aggregate.
"""

from __future__ import annotations

import argparse
import gzip
import json
import platform
from pathlib import Path

from .. import aggregate, schema
from .._log import log

RAW_SCHEMA_VERSION = "2"


def read_raw(path: Path) -> dict:
    """Load a raw trace artifact, transparently un-gzipping `.gz`."""
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as fh:
            return json.load(fh)
    return json.loads(path.read_text())


def cmd_aggregate(args: argparse.Namespace) -> None:
    for raw_path in args.raw:
        raw = read_raw(raw_path)
        if raw.get("schema_version") != RAW_SCHEMA_VERSION:
            raise SystemExit(
                f"{raw_path}: raw schema_version={raw.get('schema_version')!r}, "
                f"harness expects {RAW_SCHEMA_VERSION!r}"
            )
        # Set or backfill the machine name without re-running: --machine wins; else
        # keep what the run recorded; else fall back to the hostname.
        machine = raw.setdefault("machine", {})
        if args.machine:
            machine["host"] = args.machine
        elif not machine.get("host"):
            machine["host"] = platform.node() or "unknown"
            log(
                f"  {raw_path.name}: no machine.host in raw → {machine['host']!r} "
                f"(override with --machine)"
            )
        results = aggregate.build(raw)
        schema.validate_results(results)
        out_path = raw_path.parent / f"{raw['backend']}-results.json"
        out_path.write_text(json.dumps(results, indent=2))
        log(f"re-aggregated {raw_path} → {out_path}  ({len(results['runs'])} runs)")
