"""`bench check` — conformance-check a built backend against the contract.

Runs `version` / `providers` / one brain-check `run` on the first variant and
schema-checks each output. A backend that doesn't pass `check` is not done.
"""

from __future__ import annotations

import argparse
import json
import subprocess

from .. import config, metrics, registry
from .._log import log
from ..spawn import run as spawn_run
from ..tasks import load as load_tasks


def cmd_check(args: argparse.Namespace) -> None:
    backend = config.load_backend(args.backend)
    log(f"backend {backend.key!r} → {' '.join(backend.cmd)}")

    ver = subprocess.run([*backend.cmd, "version"], capture_output=True, text=True)
    try:
        keys = list(json.loads(ver.stdout))
    except json.JSONDecodeError as err:
        raise SystemExit(f"`version` stdout is not JSON:\n{ver.stdout[:400]!r}") from err
    log(f"✓ version is JSON ({', '.join(keys[:6])}…)")

    variants = registry.variants(args.models, backend.key)
    if not variants:
        raise SystemExit(f"no {backend.key!r} variants under {args.models}")
    v = variants[0]
    eps = registry.providers(backend, v.model_path)
    log(f"✓ providers for {v.model}/{v.model_path.name}: {eps}")

    gate = [t for t in load_tasks(args.tasks) if t.role == "gate"]
    # spawn_run schema-checks the events object before returning it.
    result = spawn_run(
        backend.cmd, model_path=v.model_path, quant=v.quant, ep=eps[0], task=gate[0].spec, iters=1
    )
    if result.events is None:
        raise SystemExit(f"`run` produced no valid events: {result.error}")
    completion = metrics.completions(result.events)[0]
    log(
        f"✓ events valid on {eps[0]} ({result.events['device']}); "
        f"healthy={result.events['healthy']}"
    )
    log(f"  {gate[0].name} → {completion!r}")
