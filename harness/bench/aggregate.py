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

Sample pools per number:

  • sweep/probe points — each point carries its own adaptive repeats, made in
    process by the exe; here they reduce to p50/min/max (+ n_reps).
  • job timing (ttft/decode_tps/prefill_tps/completion) — every iteration of
    every job spawn, [p50, max].
  • job memory and the warm load phases — the job's S spawns, [p50, max].
  • cold_start — the genuine first touch of the model on this machine, n=1.

`*_vram` collapses to null when not measurable for the provider (a CPU EP has no
VRAM).
"""

from __future__ import annotations

from statistics import median

from . import memory, metrics
from .spawn import SpawnResult

MAX_SAMPLE_COMPLETIONS = 2

NS_PER_S = 1e9
NS_PER_MS = 1e6

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


def _spread(values: list[float], prefix: str) -> dict:
    """{<prefix>_p50, <prefix>_min, <prefix>_max} over a point's repeats."""
    return {
        f"{prefix}_p50": round(median(values), 2),
        f"{prefix}_min": round(min(values), 2),
        f"{prefix}_max": round(max(values), 2),
    }


def _repeat_seconds(repeats: list[dict]) -> list[float]:
    return [(r["end_ns"] - r["start_ns"]) / NS_PER_S for r in repeats]


def _all_traces(raw: dict):
    for p in raw.get("probes") or []:
        yield p["trace"]
    for cell in raw["cells"]:
        yield from cell["gate_spawns"]
        if cell["sweep"].get("trace"):
            yield cell["sweep"]["trace"]
        yield from cell["job"]["spawns"]


def sampling_sources(raw: dict) -> dict:
    """The *run* box's sampling capabilities — {os, nvml, drm}.

    These must come off the raw artifact, never the current host: aggregation is a
    pure function of the raw, and reading the aggregating box instead would
    silently rewrite vram_method whenever a raw is re-aggregated elsewhere (a Mac
    raw re-aggregated on a Linux box would read "n/a")."""
    os = raw["machine"]["os"]
    if recorded := raw.get("sampling"):
        return {"os": os, "nvml": recorded["nvml"], "drm": recorded.get("drm", False)}
    return {
        "os": os,
        "nvml": any(s["vram"] > 0 for t in _all_traces(raw) for s in t["samples"]),
        "drm": False,
    }


def _saw_vram(traces: list[Trace]) -> bool:
    return any(s["vram"] > 0 for sp in traces for s in sp["samples"])


def vram_method(traces: list[Trace], sources: dict, provider: str) -> str:
    """nvml / drm / unified / n/a for a run, from the run box's sources, the
    provider, and whether this run actually put anything on the GPU.

    `drm` is DRM fdinfo's device-local pool — a discrete card's VRAM, or an APU's
    BIOS carve-out, which is reserved before boot and so is not the host RAM that
    lands in RSS.

    A CPU EP has no device pool *by definition*, and saying so takes an explicit
    check rather than trusting the samples: loading a model on `cpu` still brings
    the GPU backend up, and its idle bookkeeping (12 KB on the Ryzen APU) is enough
    to read as "device memory seen" and report a measured 0.0 where the honest
    answer is "not applicable"."""
    if sources["os"] == "macos":
        return "unified"
    if provider == "cpu" or not _saw_vram(traces):
        return "n/a"
    if sources["nvml"]:
        return "nvml"
    return "drm" if sources.get("drm") else "n/a"


# ── probes ────────────────────────────────────────────────────────────────────


def probe_result(provider: str, trace: Trace) -> dict:
    """One results probe entry from one probe trace. Throughputs derive from the
    declared work: GEMM moves 2·m·n·k FLOPs per repeat; a copy moves its payload
    once for h2d/d2h and reads+writes it for d2d (so ×2 — the STREAM convention
    for on-device traffic)."""
    ev = trace["events"]
    if not ev:
        return {"provider": provider, "device": "unknown", "status": "errored",
                "gemm": [], "copy": []}
    gemm = []
    for g in ev["gemm"]:
        tflops = [2 * g["m"] * g["n"] * g["k"] / s / 1e12 for s in _repeat_seconds(g["repeats"])]
        gemm.append({"m": g["m"], "n": g["n"], "k": g["k"], "dtype": g["dtype"],
                     "tflops_p50": round(median(tflops), 2), "n_reps": len(tflops)})
    copy = []
    for c in ev["copy"]:
        traffic = c["bytes"] * (2 if c["kind"] == "d2d" else 1)
        gbs = [traffic / s / 1e9 for s in _repeat_seconds(c["repeats"])]
        copy.append({"kind": c["kind"], "bytes": c["bytes"],
                     "gbs_p50": round(median(gbs), 2), "n_reps": len(gbs)})
    return {"provider": provider, "device": ev["device"], "status": "ok",
            "gemm": gemm, "copy": copy}


# ── sweeps ────────────────────────────────────────────────────────────────────


def sweep_result(cell_sweep: dict) -> dict:
    """Aggregated sweep points from the sweep trace. Points survive a non-ok
    status — whatever completed before a failure still informs the fit.
    Prefill chunks pass through as the marginal cost curve (exact, single
    pass); decode points reduce to their tps spread."""
    ev = (cell_sweep.get("trace") or {}).get("events")
    prefill, decode = [], []
    if ev:
        for p in ev["prefill_chunks"]:
            prefill.append({"context": p["context_size"], "tokens": p["tokens_count"],
                            "ms": round((p["end_ns"] - p["start_ns"]) / NS_PER_MS, 2)})
        for d in ev["decode_points"]:
            tps = [
                (len(r["token_ns"]) - 1) / ((r["token_ns"][-1] - r["token_ns"][0]) / NS_PER_S)
                for r in d["repeats"]
            ]
            decode.append({"kv_fill": d["kv_fill"], "tokens": d["tokens"],
                           **_spread(tps, "tps"), "n_reps": len(tps)})
    return {"status": cell_sweep["status"], "prefill": prefill, "decode": decode}


# ── the job ───────────────────────────────────────────────────────────────────


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
    return seen or [""]  # schema requires ≥1; an all-failed job shouldn't reach here


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
        return stat(vals) if method in ("nvml", "drm") else None

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


def job_result(cell_job: dict, *, method: str, cold_start_ms: float | None) -> dict:
    """The validation job entry. Metrics only when the job scored (`ok`) — a
    too-slow or errored job keeps its status and task name, nothing invented."""
    out = {"status": cell_job["status"], "task": cell_job["task"]}
    if cell_job["status"] == "ok":
        tr = task_result(cell_job["task"], cell_job["spawns"], method=method,
                         cold_start_ms=cold_start_ms)
        out["metrics"] = tr["metrics"]
        out["memory"] = tr["memory"]
        out["sample_completions"] = tr["sample_completions"]
    return out


# ── assembly ──────────────────────────────────────────────────────────────────


def _geometry(traces: list[Trace]) -> dict | None:
    """The geometry block from the first spawn that reported one — carried
    verbatim; the runtime is the authority."""
    for t in traces:
        if t["events"] and "geometry" in t["events"]:
            return t["events"]["geometry"]
    return None


def _run_from_cell(cell: dict, sources: dict) -> dict:
    """One results `run` from one raw cell. The cell carries the *structure* and the
    runtime decisions (gate verdict, sweep/job statuses, the cold-load value to
    attribute); everything numeric is (re)derived from the traces here."""
    sweep_traces = [cell["sweep"]["trace"]] if cell["sweep"].get("trace") else []
    all_traces = cell["gate_spawns"] + sweep_traces + cell["job"]["spawns"]
    method = vram_method(all_traces, sources, cell["provider"])
    device = next((s["events"]["device"] for s in all_traces if s["events"]), "unknown")

    run = {
        "provider": cell["provider"],
        "device": device,
        "model": cell["model"],
        "quant": cell["quant"],
        "healthy": cell["healthy"],
        "vram_method": method,
        "geometry": _geometry(sweep_traces + cell["job"]["spawns"] + cell["gate_spawns"]),
        "sweep": sweep_result(cell["sweep"]),
        "job": job_result(cell["job"], method=method, cold_start_ms=cell.get("cold_ms")),
    }
    if not cell["healthy"] and cell.get("reason"):
        run["unhealthy_reason"] = cell["reason"]
    return run


def build(raw: dict) -> dict:
    """Raw intermediaries → results object. Pure; the single
    aggregation entrypoint shared by live `run` and `bench aggregate`."""
    sources = sampling_sources(raw)
    return {
        "schema_version": "2",
        "backend": raw["backend"],
        "machine": raw["machine"],
        "job_spawns": raw["job_spawns"],
        "probes": [probe_result(p["provider"], p["trace"]) for p in raw.get("probes") or []],
        "runs": [_run_from_cell(cell, sources) for cell in raw["cells"]],
    }
