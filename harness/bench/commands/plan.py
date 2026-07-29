"""`bench plan` — enumerate the cells without running them.

The cheap smoke test that `backend.toml`, `models.yaml`, and each artifact's
`providers` line up before committing to a multi-hour run.
"""

from __future__ import annotations

import argparse

from .. import config, registry
from .._log import log
from ..tasks import load as load_tasks


def cmd_plan(args: argparse.Namespace) -> None:
    backend = config.load_backend(args.backend)
    tasks = load_tasks(args.tasks)
    gate = [t.name for t in tasks if t.role == "gate"]
    timed = [t.name for t in tasks if t.role == "timed"]
    variants = registry.variants(args.models, backend.key)
    if not variants:
        raise SystemExit(f"no {backend.key!r} variants under {args.models}")
    variants = registry.select(variants, args.model)

    log(f"backend {backend.key!r} → {' '.join(backend.cmd)}")
    log(f"gate: {gate}   job: {timed}")
    cells = 0
    lanes_seen: dict[str, str] = {}
    for v in variants:
        lanes = registry.filter_lanes(registry.providers(backend, v.model_path), args.providers)
        lanes_seen.update({lane.id: lane.description for lane in lanes})
        print(f"{v.model:28} {v.quant:5}  lanes={[lane.id for lane in lanes]}")
        cells += len(lanes)
    print(
        f"\n{len(variants)} variants → {cells} cells (each: gate + sweep + job), "
        f"plus one ceiling probe per device lane:"
    )
    for lane_id, description in sorted(lanes_seen.items()):
        print(f"  {lane_id:12} {description}")
