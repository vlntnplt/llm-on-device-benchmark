"""Pool a run's raw traces into the results object.

This is the *consumer* half of the producer/consumer split. It is a
pure function of the persisted intermediaries — `build(raw)` takes the trace
artifact `bench run` writes (`<backend>-raw.json.gz`: per-spawn `events` + sample
series) and returns the validated results object. The live `run` path and the
standalone `bench aggregate` re-run go through the *same* `build`, so changing how
a metric is summarized is a re-aggregate, never a re-inference.

A *trace* is the atomic input here: `{"events": <events object | None>, "samples":
[<sample>, …]}` — exactly one spawn. Everything below reads only those two keys, so
it doesn't care whether the trace came straight off a live `SpawnResult` or was
loaded back from disk.

Where S and K stop being symmetric — each metric is `[p50, max]` (p50 ranks the
stacks, max flags instability) but over a different pool:

  • timing (ttft/decode_tps/prefill_tps/completion) — every iteration, all spawns
    → S×K samples.
  • per-task memory (prefill/decode high-water) and the warm load phases
    (model_load/context_init/warmup) — that task's S spawns.
  • cold_start — the genuine first touch of the model on this machine, n=1.

`*_vram` collapses to null when not measurable for the provider (a CPU EP has no
VRAM).
"""

from __future__ import annotations

from statistics import median

from . import memory, metrics
from .spawn import SpawnResult

MAX_SAMPLE_COMPLETIONS = 2

# A trace: {"events": dict | None, "samples": list[dict]}. The serializable slice
# of a spawn that aggregation actually consumes.
Trace = dict


def trace_of(spawn: SpawnResult) -> Trace:
    """The persistable intermediary for one spawn — events + sample series, the
    only two things aggregation reads. The control-flow fields (timed_out, error,
    cold) drove the run; they're not needed to re-aggregate."""
    return {"events": spawn.events, "samples": spawn.samples}


def stat(values: list[float | None]) -> list[float] | None:
    """[p50, max] over the non-null values, or None if there are none."""
    vals = [v for v in values if v is not None]
    if not vals:
        return None
    return [round(median(vals), 2), round(max(vals), 2)]


def _raw_samples(raw: dict):
    for cell in raw["cells"]:
        for group in [cell["gate_spawns"], *(g["spawns"] for g in cell["tasks"])]:
            for sp in group:
                yield from sp["samples"]


def sampling_sources(raw: dict) -> dict:
    """The *run* box's sampling capabilities — {os, nvml}.

    These must come off the raw artifact, never the current host: aggregation is a
    pure function of the raw, and reading the aggregating box instead would
    silently rewrite vram_method whenever a raw is re-aggregated elsewhere (a Mac
    raw re-aggregated on a Linux box would read "n/a").

    `bench run` records them (raw["sampling"]); a raw without the block is still
    aggregatable by inferring from the data: any VRAM sample > 0 means NVML
    produced it (the only vram source — GTT folds into rss). The inference only
    misreads a box whose capability went unused — exactly the case where
    vram_method collapses to n/a anyway."""
    os = raw["machine"]["os"]
    if recorded := raw.get("sampling"):
        return {"os": os, "nvml": recorded["nvml"]}
    return {"os": os, "nvml": any(s["vram"] > 0 for s in _raw_samples(raw))}


def _saw_vram(traces: list[Trace]) -> bool:
    return any(s["vram"] > 0 for sp in traces for s in sp["samples"])


def vram_method(traces: list[Trace], sources: dict) -> str:
    """nvml / unified / n/a for a run, from the run box's sources + whether this
    run actually put anything on the GPU."""
    if sources["os"] == "macos":
        return "unified"
    if sources["nvml"] and _saw_vram(traces):
        return "nvml"
    return "n/a"


def _completions(traces: list[Trace]) -> list[str]:
    seen: list[str] = []
    for sp in traces:
        if not sp["events"]:
            continue
        for text in metrics.completions(sp["events"]):
            if text not in seen:
                seen.append(text)
            if len(seen) >= MAX_SAMPLE_COMPLETIONS:
                return seen
    return seen or [""]  # schema requires ≥1; an all-failed task shouldn't reach here


def task_result(
    name: str,
    traces: list[Trace],
    *,
    method: str,
    cold_start_ms: float | None,
) -> dict:
    ok = [s for s in traces if s["events"]]
    timings = [t for s in ok for t in metrics.timing_samples(s["events"])]
    mems = [memory.attribute(s["events"], s["samples"]) for s in ok]
    loads = [metrics.load_components(s["events"]) for s in ok]

    def t(key: str) -> list:
        return [x[key] for x in timings]

    def m(key: str) -> list:
        return [x[key] for x in mems]

    def ld(key: str) -> list:
        return [x[key] for x in loads]

    def vram(vals: list) -> list | None:
        return stat(vals) if method == "nvml" else None

    return {
        "task": name,
        "metrics": {
            "cold_start_ms": [round(cold_start_ms, 2)] * 2 if cold_start_ms is not None else None,
            "model_load_ms": stat(ld("model_load_ms")),
            "context_init_ms": stat(ld("context_init_ms")),
            "warmup_ms": stat(ld("warmup_ms")),
            "ttft_ms": stat(t("ttft_ms")),
            "decode_tps": stat(t("decode_tps")),
            "prefill_tps": stat(t("prefill_tps")),
            "completion_ms": stat(t("completion_ms")),
        },
        "memory": {
            "prefill_rss_peak_mb": stat(m("prefill_rss_peak_mb")),
            "prefill_vram_peak_mb": vram(m("prefill_vram_peak_mb")),
            "decode_rss_peak_mb": stat(m("decode_rss_peak_mb")),
            "decode_vram_peak_mb": vram(m("decode_vram_peak_mb")),
            "decode_rss_sustained_mb": stat(m("decode_rss_sustained_mb")),
            "decode_vram_sustained_mb": vram(m("decode_vram_sustained_mb")),
        },
        "sample_completions": _completions(traces),
    }


def run_result(
    *,
    model: str,
    quant: str,
    provider: str,
    healthy: bool,
    all_traces: list[Trace],
    task_results: list[dict],
    unhealthy_reason: str | None,
    sources: dict,
    timed_out_tasks: list[str] | None = None,
    errored_tasks: list[str] | None = None,
) -> dict:
    """Assemble one results `run` (one model/variant/provider). `sources` is the
    run box's sampling_sources(raw) — methods derive from it, never from the
    aggregating host."""
    method = vram_method(all_traces, sources)
    device = next((s["events"]["device"] for s in all_traces if s["events"]), "unknown")
    run = {
        "provider": provider,
        "device": device,
        "model": model,
        "quant": quant,
        "healthy": healthy,
        "vram_method": method,
        "tasks": task_results,
    }
    if not healthy and unhealthy_reason:
        run["unhealthy_reason"] = unhealthy_reason
    if timed_out_tasks:  # cells too slow to score within the budget
        run["timed_out_tasks"] = sorted(timed_out_tasks)
    if errored_tasks:  # attempted but produced no sample (crash/OOM) — not slowness
        run["errored_tasks"] = sorted(errored_tasks)
    return run


def _run_from_cell(cell: dict, sources: dict) -> dict:
    """One results `run` from one raw cell. The cell carries the *structure* and the
    runtime decisions (gate verdict, which tasks were too slow, the cold-load value
    to attribute); everything numeric is (re)derived from the traces here."""
    gate = cell["gate_spawns"]
    task_groups = cell["tasks"]  # every task that ran spawns, in run order
    too_slow = set(cell.get("timed_out_tasks") or [])
    errored = set(cell.get("errored_tasks") or [])
    unusable = too_slow | errored
    all_traces = gate + [s for g in task_groups for s in g["spawns"]]

    method = vram_method(all_traces, sources)

    # The cold first-touch load is attributed once, to the first *scored* task — the
    # producer stamped `cold_ms` on the owning cell (null elsewhere).
    scored = [g for g in task_groups if g["task"] not in unusable]
    cold_for = scored[0]["task"] if scored else None
    task_results = [
        task_result(
            g["task"],
            g["spawns"],
            method=method,
            cold_start_ms=cell.get("cold_ms") if g["task"] == cold_for else None,
        )
        for g in scored
    ]
    return run_result(
        model=cell["model"],
        quant=cell["quant"],
        provider=cell["provider"],
        healthy=cell["healthy"],
        all_traces=all_traces,
        task_results=task_results,
        unhealthy_reason=cell.get("reason"),
        sources=sources,
        timed_out_tasks=sorted(too_slow),
        errored_tasks=sorted(errored),
    )


def build(raw: dict) -> dict:
    """Raw intermediaries → results object. Pure; the single
    aggregation entrypoint shared by live `run` and `bench aggregate`."""
    sources = sampling_sources(raw)
    return {
        "schema_version": "1",
        "backend": raw["backend"],
        "machine": raw["machine"],
        "iters": raw["iters"],
        "spawns": raw["spawns"],
        "runs": [_run_from_cell(cell, sources) for cell in raw["cells"]],
    }
