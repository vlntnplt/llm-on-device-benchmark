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
    providers: set[str] = set()
    for v in variants:
        eps = registry.providers(backend, v.model_path)
        if args.providers:
            eps = [e for e in eps if e in args.providers]
        providers.update(eps)
        print(f"{v.model:28} {v.quant:5}  providers={eps}")
        cells += len(eps)
    print(
        f"\n{len(variants)} variants → {cells} cells (each: gate + sweep + job), "
        f"plus one ceiling probe per provider ({sorted(providers)})"
    )
