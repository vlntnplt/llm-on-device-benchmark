"""`bench run` — the benchmark proper.

Per `(model, variant, provider)`: gate on the brain-check, then time the healthy
ones (S spawns × K iters). Persists the raw traces, then aggregates them to
[p50, max] results through the same `aggregate.build` that `bench aggregate` uses.
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

# Below ~4 tok/s a stack is slower than a person reads — effectively unusable. The
# summarize family grows monotonically, so once the cheapest task is this slow (or
# times out), the costlier ones can only be worse: skip them, mark unusable.
UNUSABLE_TPS = 4


def _spawn(
    backend: config.Backend,
    v: registry.Variant,
    ep: str,
    task: Task,
    *,
    iters: int,
    cold: bool = False,
    deadline_ms: int | None = None,
    backstop_s: float | None = None,
) -> spawn.SpawnResult:
    return spawn.run(
        backend.cmd,
        model_path=v.model_path,
        quant=v.quant,
        ep=ep,
        task=task.spec,
        iters=iters,
        cold=cold,
        deadline_ms=deadline_ms,
        backstop_s=backstop_s,
    )


def _decode_tps(result: spawn.SpawnResult) -> float | None:
    """Median decode tok/s over a spawn's iterations — the progress heartbeat, not
    the aggregated stat."""
    if not result.events:
        return None
    vals = [t["decode_tps"] for t in metrics.timing_samples(result.events) if t["decode_tps"]]
    return median(vals) if vals else None


def _task_cost(task: Task) -> tuple[int, int]:
    """Cheapest-first ordering key: prompt budget, then total tokens to generate."""
    spec = task.spec
    return (spec.get("max_context_length", 0), sum(m.get("nb_tokens", 0) for m in spec["messages"]))


def cmd_run(args: argparse.Namespace) -> None:
    backend = config.load_backend(args.backend)
    tasks = load_tasks(args.tasks)
    gate = [t for t in tasks if t.role == "gate"]
    timed = [t for t in tasks if t.role == "timed"]
    variants = registry.variants(args.models, backend.key)
    if not variants:
        raise SystemExit(f"no {backend.key!r} variants under {args.models}")

    # Pre-resolve the (variant, provider) cells so progress can show [i/N]; this
    # asks each artifact's `providers` exactly once.
    cells: list[tuple[registry.Variant, str]] = []
    for v in variants:
        eps = registry.providers(backend, v.model_path)
        if args.providers:
            eps = [e for e in eps if e in args.providers]
        cells += [(v, ep) for ep in eps]
    deadline_ms = args.max_ms or None  # soft per-spawn time-box
    backstop_s = args.backstop_ms / 1000  # hard kill floor for one runaway iteration
    log(
        f"{len(cells)} cells (gate + {len(timed)} timed, {args.spawns} spawns x {args.iters} iters)"
        + (
            f"; deadline {deadline_ms / 1000:.0f}s, backstop {backstop_s:.0f}s"
            if deadline_ms
            else ""
        )
    )

    touched: set[Path] = set()  # model files already loaded once on this machine
    cold_load: dict[Path, float] = {}  # genuine first-touch load (cold page cache)
    cold_used: set[Path] = set()  # cold_start already attributed to a task
    overruns: list[str] = []  # cells whose rendered prompt exceeded its context
    raw_cells: list[dict] = []  # raw per-cell traces → persisted, then aggregated

    for idx, (v, ep) in enumerate(cells, 1):
        head = f"[{idx}/{len(cells)}] {v.model} {v.quant} {ep}"
        # 1. brain-check gate (once, iters=1).
        gate_spawns = []
        for gt in gate:
            is_cold = v.model_path not in touched
            touched.add(v.model_path)
            s = _spawn(backend, v, ep, gt, iters=1, cold=is_cold, backstop_s=backstop_s)
            if is_cold and s.events:
                cold_load[v.model_path] = metrics.load_ms(s.events)
            gate_spawns.append((gt, s))
        healthy = all(s.healthy for _, s in gate_spawns)
        marks = "".join("✓" if s.healthy else "✗" for _, s in gate_spawns)
        per_task: dict[str, list[spawn.SpawnResult]] = {}
        timed_out_tasks: list[str] = []  # too slow: backstop timeout or below the floor
        errored_tasks: list[str] = []  # attempted but produced no sample (crash/OOM)
        reason = None

        if not healthy:
            reason = "; ".join(
                f"{gt.name}: {s.error or 'expect failed'}" for gt, s in gate_spawns if not s.healthy
            )
            log(f"{head}  gate {marks} — UNHEALTHY ({reason}); skipping timed")
        else:
            log(f"{head}  gate {marks}")
            # 2. timed tasks, cheapest-first: up to S spawns × ≤K iters each. Two ways
            # to stop wasting time on a slow cell: a slow/killed first
            # spawn is re-grindless, so stop re-spawning it; and once any task is
            # unusable (no sample, or below the floor) the costlier tasks can only be
            # worse — don't run them, just mark them unusable.
            cell_too_slow = False
            fail_bucket = None  # which list a skipped larger task inherits
            for tk in sorted(timed, key=_task_cost):
                if cell_too_slow:
                    fail_bucket.append(tk.name)
                    log(f"    {tk.name}: skipped — cell already unusable for a smaller task")
                    continue
                sp = []
                for j in range(args.spawns):
                    s = _spawn(
                        backend,
                        v,
                        ep,
                        tk,
                        iters=args.iters,
                        deadline_ms=deadline_ms,
                        backstop_s=backstop_s,
                    )
                    sp.append(s)
                    d = _decode_tps(s)
                    note = (
                        f"{d:.0f} tok/s"
                        if d
                        else ("⏱ too slow" if s.timed_out else f"<{s.error}>" if s.error else "—")
                    )
                    iters_done = len(s.events["iterations"]) if s.events else 0
                    tail = f" ({iters_done}/{args.iters} iters)" if s.truncated else ""
                    log(f"    {tk.name} {j + 1}/{args.spawns}: decode {note}{tail}")
                    if j == 0 and (s.timed_out or s.truncated or not s.events):
                        log(f"    {tk.name}: bad first spawn — skipping remaining spawns")
                        break
                per_task[tk.name] = sp
                # Flag (loudly) any cell whose rendered prompt overran its context,
                # from the exe's own token counts; never trim — adjust by hand.
                budget = tk.spec.get("max_context_length")
                first = next((s for s in sp if s.events), None)
                if budget and first and (peak := metrics.peak_context(first.events)) > budget:
                    log(f"    ⚠️  {tk.name}: sequence {peak} tok > max_context_length {budget}")
                    overruns.append(f"{v.model} {v.quant} {ep} · {tk.name}: {peak} > {budget}")
                # Verdict for this task → can the cell still serve bigger ones? No
                # sample at all splits two ways: a backstop timeout is genuinely
                # too slow; anything else (a spawn that died producing nothing) is
                # an error, not slowness — keep them apart so the report can too.
                if not any(s.events for s in sp):
                    cell_too_slow = True
                    if any(s.timed_out for s in sp):
                        fail_bucket = timed_out_tasks
                        timed_out_tasks.append(tk.name)
                        log(f"    {tk.name}: ⏱ too slow — skipping any larger task")
                    else:
                        fail_bucket = errored_tasks
                        errored_tasks.append(tk.name)
                        err = next((s.error for s in sp if s.error), None)
                        log(
                            f"    {tk.name}: ✗ no sample{f' ({err})' if err else ''}"
                            " — skipping any larger task"
                        )
                else:
                    best = max((_decode_tps(s) or 0.0) for s in sp)
                    if best < UNUSABLE_TPS:  # F-grade: bigger tasks are hopeless
                        cell_too_slow = True
                        fail_bucket = timed_out_tasks
                        log(
                            f"    {tk.name}: {best:.1f} tok/s < {UNUSABLE_TPS:g} — "
                            f"larger tasks unusable, skipping them"
                        )
            if timed_out_tasks:
                log(f"    ⏱ too slow: {', '.join(timed_out_tasks)}")
            if errored_tasks:
                log(f"    ✗ errored: {', '.join(errored_tasks)}")

        # Record the cell's raw traces — events + sample series per spawn. The
        # structure and runtime verdicts (gate health, which tasks were too slow,
        # the cold value to attribute) live here; aggregate.build (re)derives every
        # number from the traces.
        ran = [tk for tk in sorted(timed, key=_task_cost) if tk.name in per_task]
        cell = {
            "model": v.model,
            "quant": v.quant,
            "provider": ep,
            "healthy": healthy,
            "reason": reason,
            "timed_out_tasks": timed_out_tasks,
            "errored_tasks": errored_tasks,
            "cold_ms": None,
            "gate_spawns": [aggregate.trace_of(s) for _, s in gate_spawns],
            "tasks": [
                {"task": tk.name, "spawns": [aggregate.trace_of(s) for s in per_task[tk.name]]}
                for tk in ran
            ],
        }
        # The one genuine cold first-touch is attributed once, to the first scored
        # task of the first cell that has one (cold_used is run-global).
        if (
            v.model_path in cold_load
            and v.model_path not in cold_used
            and any(tk.name not in timed_out_tasks and tk.name not in errored_tasks for tk in ran)
        ):
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
        "iters": args.iters,
        "spawns": args.spawns,
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
