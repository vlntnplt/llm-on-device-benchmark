"""`bench run` — the benchmark proper.

Per provider: one device-ceiling probe. Per `(model, variant, provider)`: gate
on the brain-check, then measure the healthy ones — one sweep spawn (the exe
repeats each point adaptively in-process) and S job spawns of the single
validation task. Persists the raw traces, then aggregates them through the same
`aggregate.build` that `bench aggregate` uses.
"""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path
from statistics import median

from .. import aggregate, config, machine, metrics, registry, sampling, schema, spawn
from .._log import log
from ..tasks import Task
from ..tasks import load as load_tasks
from .aggregate import RAW_SCHEMA_VERSION

# Below ~4 tok/s a stack is slower than a person reads — effectively unusable.
UNUSABLE_TPS = 4

PROBE_BACKSTOP_S = 300.0

# The sweep's hard kill is a hang guard, never a routine stop (a kill loses
# every completed point). Budgeted sweeps get the budget plus a tail — one
# in-flight chunk and the decode/memory ladders run past the budget by design.
SWEEP_HANG_BACKSTOP_S = 3600.0
SWEEP_TAIL_S = 300.0


def _decode_tps(result: spawn.SpawnResult) -> float | None:
    """Median decode tok/s over a spawn's iterations — the progress heartbeat, not
    the aggregated stat."""
    if not result.events:
        return None
    vals = [t["decode_tps"] for t in metrics.timing_samples(result.events) if t["decode_tps"]]
    return median(vals) if vals else None


def _gate(backend, v, ep, gate_tasks: list[Task], *, touched, cold_load, backstop_s):
    """Run each gate task once (the catalog defines one multi-turn brain-check,
    so this is one spawn); track the genuine cold first-touch. Unsampled —
    memory is the job's to measure."""
    gate_spawns = []
    for gt in gate_tasks:
        is_cold = v.model_path not in touched
        touched.add(v.model_path)
        s = spawn.run(backend.cmd, model_path=v.model_path, quant=v.quant, ep=ep,
                      task=gt.spec, iters=1, cold=is_cold, backstop_s=backstop_s)
        if is_cold and s.events:
            cold_load[v.model_path] = metrics.load_ms(s.events)
        gate_spawns.append((gt, s))
    return gate_spawns


def _sweep(backend, v, ep, *, deadline_ms, backstop_s) -> tuple[str, spawn.SpawnResult]:
    """One sweep spawn → (status, result). A killed sweep is `too_slow`, a spawn
    that died emitting nothing is `errored`; partial points still count as ok —
    each point carries its own truth."""
    s = spawn.sweep(backend.cmd, model_path=v.model_path, quant=v.quant, ep=ep,
                    deadline_ms=deadline_ms, backstop_s=backstop_s)
    if s.events:
        return "ok", s
    return ("too_slow" if s.timed_out else "errored"), s


def _job(backend, v, ep, task: Task, *, spawns, iters, deadline_ms, backstop_s):
    """S spawns of the validation job → (status, results) — the only sampled
    spawns (memory is measured here and nowhere else). A bad first spawn is
    not re-ground; a scored job below the usable floor is too_slow, kept apart
    from errored (crash/OOM) so the report can be too."""
    sp: list[spawn.SpawnResult] = []
    for j in range(spawns):
        s = spawn.run(backend.cmd, model_path=v.model_path, quant=v.quant, ep=ep,
                      task=task.spec, iters=iters, deadline_ms=deadline_ms,
                      backstop_s=backstop_s, sample=True)
        sp.append(s)
        d = _decode_tps(s)
        note = (f"{d:.0f} tok/s" if d
                else ("⏱ too slow" if s.timed_out else f"<{s.error}>" if s.error else "—"))
        iters_done = len(s.events["iterations"]) if s.events else 0
        tail = f" ({iters_done}/{iters} iters)" if s.truncated else ""
        log(f"    job {task.name} {j + 1}/{spawns}: decode {note}{tail}")
        if j == 0 and (s.timed_out or s.truncated or not s.events):
            log("    job: bad first spawn — skipping remaining spawns")
            break

    if not any(s.events for s in sp):
        status = "too_slow" if any(s.timed_out for s in sp) else "errored"
    elif max((_decode_tps(s) or 0.0) for s in sp) < UNUSABLE_TPS:
        status = "too_slow"
    else:
        status = "ok"
    return status, sp


def cmd_run(args: argparse.Namespace) -> None:
    backend = config.load_backend(args.backend)
    tasks = load_tasks(args.tasks)
    gate = [t for t in tasks if t.role == "gate"]
    timed = [t for t in tasks if t.role == "timed"]
    if len(timed) != 1:
        raise SystemExit(
            f"the task catalog must define exactly one validation job, found "
            f"{[t.name for t in timed]}"
        )
    job_task = timed[0]
    variants = registry.variants(args.models, backend.key)
    if not variants:
        raise SystemExit(f"no {backend.key!r} variants under {args.models}")
    variants = registry.select(variants, args.model)

    # Pre-resolve the (variant, provider) cells so progress can show [i/N]; this
    # asks each artifact's `providers` exactly once.
    cells: list[tuple[registry.Variant, str]] = []
    for v in variants:
        eps = registry.providers(backend, v.model_path)
        if args.providers:
            eps = [e for e in eps if e in args.providers]
        cells += [(v, ep) for ep in eps]
    deadline_ms = args.max_ms or None  # soft per-job-spawn time-box
    backstop_s = args.backstop_ms / 1000  # hard kill floor for one runaway iteration
    sweep_deadline_ms = args.sweep_ms or None
    sweep_backstop_s = (args.sweep_ms / 1000 + SWEEP_TAIL_S if args.sweep_ms
                        else SWEEP_HANG_BACKSTOP_S)
    log(
        f"{len(cells)} cells (gate + sweep + job '{job_task.name}' × {args.spawns} spawns); "
        f"probe per provider"
        + (f"; sweep deadline {sweep_deadline_ms / 1000:.0f}s" if sweep_deadline_ms else "")
    )

    touched: set[Path] = set()  # model files already loaded once on this machine
    cold_load: dict[Path, float] = {}  # genuine first-touch load (cold page cache)
    cold_used: set[Path] = set()  # cold_start already attributed
    overruns: list[str] = []  # cells whose rendered prompt exceeded the job's context
    probes: list[dict] = []  # one ceiling probe per provider
    probed: set[str] = set()
    raw_cells: list[dict] = []  # raw per-cell traces → persisted, then aggregated

    for idx, (v, ep) in enumerate(cells, 1):
        head = f"[{idx}/{len(cells)}] {v.model} {v.quant} {ep}"

        # 0. one ceiling probe per provider, ahead of its first cell.
        if ep not in probed:
            probed.add(ep)
            p = spawn.probe(backend.cmd, ep=ep, backstop_s=PROBE_BACKSTOP_S)
            probes.append({"provider": ep, "trace": aggregate.trace_of(p)})
            log(f"probe {ep}: {'ok' if p.events else f'✗ {p.error}'}")

        # 1. brain-check gate (once, iters=1).
        gate_spawns = _gate(backend, v, ep, gate, touched=touched, cold_load=cold_load,
                            backstop_s=backstop_s)
        healthy = all(s.healthy for _, s in gate_spawns)
        marks = "".join("✓" if s.healthy else "✗" for _, s in gate_spawns)

        sweep_status, sweep_res = "skipped", None
        job_status, job_spawns = "skipped", []
        reason = None
        if not healthy:
            reason = "; ".join(
                f"{gt.name}: {s.error or 'expect failed'}" for gt, s in gate_spawns if not s.healthy
            )
            log(f"{head}  gate {marks} — UNHEALTHY ({reason}); skipping sweep + job")
        else:
            log(f"{head}  gate {marks}")
            # 2. the sweep — the exe measures its points with adaptive repeats.
            sweep_status, sweep_res = _sweep(backend, v, ep, deadline_ms=sweep_deadline_ms,
                                             backstop_s=sweep_backstop_s)
            if sweep_status == "ok":
                pts = sweep_res.events
                depth = sum(p["tokens_count"] for p in pts["prefill_chunks"])
                log(f"    sweep: prefill to {depth} tokens "
                    f"({len(pts['prefill_chunks'])} chunks) + "
                    f"{len(pts['decode_points'])} decode points")
            else:
                log(f"    sweep: ✗ {sweep_status} ({sweep_res.error})")
            # 3. the validation job.
            job_status, job_spawns = _job(backend, v, ep, job_task, spawns=args.spawns,
                                          iters=args.iters, deadline_ms=deadline_ms,
                                          backstop_s=backstop_s)
            # Flag (loudly) any cell whose rendered prompt overran its context,
            # from the exe's own token counts; never trim — adjust by hand.
            budget = job_task.spec.get("max_context_length")
            first = next((s for s in job_spawns if s.events), None)
            if budget and first and (peak := metrics.peak_context(first.events)) > budget:
                log(f"    ⚠️  {job_task.name}: sequence {peak} tok > max_context_length {budget}")
                overruns.append(f"{v.model} {v.quant} {ep}: {peak} > {budget}")

        cell = {
            "model": v.model,
            "quant": v.quant,
            "provider": ep,
            "healthy": healthy,
            "reason": reason,
            "cold_ms": None,
            "gate_spawns": [aggregate.trace_of(s) for _, s in gate_spawns],
            "sweep": {"status": sweep_status,
                      "trace": aggregate.trace_of(sweep_res) if sweep_res else None},
            "job": {"task": job_task.name, "status": job_status,
                    "spawns": [aggregate.trace_of(s) for s in job_spawns]},
        }
        # The one genuine cold first-touch is attributed once, to the first cell
        # whose job scored (cold_used is run-global).
        if v.model_path in cold_load and v.model_path not in cold_used and job_status == "ok":
            cell["cold_ms"] = cold_load[v.model_path]
            cold_used.add(v.model_path)
        raw_cells.append(cell)

    raw = {
        "schema_version": RAW_SCHEMA_VERSION,
        "backend": backend.key,
        "machine": machine.info(args.machine),
        # The run box's sampling sources, recorded so re-aggregating this raw on a
        # different box reproduces the same vram_method (aggregate.sampling_sources).
        "sampling": {"nvml": sampling.NVML_AVAILABLE, "drm": sampling.DRM_AVAILABLE},
        "job_spawns": args.spawns,
        "job_iters": args.iters,
        "probes": probes,
        "cells": raw_cells,
    }
    args.out.mkdir(parents=True, exist_ok=True)
    raw_path = args.out / f"{backend.key}-raw.json.gz"
    with gzip.open(raw_path, "wt", encoding="utf-8") as fh:
        json.dump(raw, fh)
    log(f"wrote {raw_path}  (raw traces, {len(raw_cells)} cells)")

    results = aggregate.build(raw)
    schema.validate_results(results)
    out_path = args.out / f"{backend.key}-results.json"
    out_path.write_text(json.dumps(results, indent=2))
    log(f"wrote {out_path}  ({len(results['runs'])} runs)")

    if overruns:
        log("")
        log(
            f"⚠️  {len(overruns)} prompt overrun(s) — these cells exceeded max_context_length; "
            "inference may have truncated. Trim the corpus or raise the task's budget:"
        )
        for line in overruns:
            log(f"    {line}")
