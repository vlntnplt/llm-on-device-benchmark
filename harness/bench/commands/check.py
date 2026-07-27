"""`bench check` — conformance-check a built backend against the contract.

Runs `version` / `providers`, one ceiling `probe`, a minimal `sweep` (a 1 ms
deadline keeps it to the first point of each kind — the exe's first-point
guarantee makes that a complete, schema-valid object), and one brain-check
`run` on the first variant; schema-checks each output. A backend that doesn't
pass `check` is not done.
"""

from __future__ import annotations

import argparse
import json
import subprocess

from .. import config, metrics, registry
from .._log import log
from ..spawn import probe as spawn_probe
from ..spawn import run as spawn_run
from ..spawn import sweep as spawn_sweep
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

    # spawn_* schema-check every events object before returning it.
    pr = spawn_probe(backend.cmd, ep=eps[0], backstop_s=300)
    if pr.events is None:
        raise SystemExit(f"`probe` produced no valid events: {pr.error}")
    best = max(
        2 * g["m"] * g["n"] * g["k"] / min((r["end_ns"] - r["start_ns"]) for r in g["repeats"])
        * 1e9 / 1e12
        for g in pr.events["gemm"]
    )
    log(f"✓ probe valid on {eps[0]} ({pr.events['device']}); best gemm {best:.1f} TFLOP/s")

    sw = spawn_sweep(backend.cmd, model_path=v.model_path, quant=v.quant, ep=eps[0],
                     deadline_ms=1, backstop_s=1800)
    if sw.events is None:
        raise SystemExit(f"`sweep` produced no valid events: {sw.error}")
    g = sw.events["geometry"]
    kinds: dict[str, int] = {}
    for layer in g["layers"]:
        kinds[layer["kind"]] = kinds.get(layer["kind"], 0) + 1
    log(
        f"✓ sweep valid ({len(sw.events['prefill_chunks'])} prefill chunks / "
        f"{len(sw.events['decode_points'])} decode points); geometry: "
        f"{g['n_layer']} layers {kinds}, n_ubatch={g['context']['n_ubatch']}"
    )

    gate = [t for t in load_tasks(args.tasks) if t.role == "gate"]
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
