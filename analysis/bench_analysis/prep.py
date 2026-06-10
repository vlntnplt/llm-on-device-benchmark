"""Derived frames for the comparison notebook.

`load` builds the raw tidy frame; this module derives the views the notebook
plots. Pure pandas, no plotting, unit-tested — so the notebook stays a
disposable arrangement of tested pieces.
"""

from __future__ import annotations

import re

import pandas as pd

# Context ladder small→medium→large, not alphabetical.
TASK_LADDER = ["summarize-small", "summarize-medium", "summarize-large"]

# Stacked-chart segment labels, bottom→top draw order. `charts` keys its
# palettes off these, so the labels live in exactly one place.
TIME_PHASES = ["model load", "context init", "warmup", "prefill", "decode"]
MEMORY_PHASES = ["RAM", "VRAM", "transient peak"]

FAIL_LABELS = {"too_slow": "too slow", "errored": "errored"}


# Vendor noise in cpu/gpu strings: parentheticals ("(RADV PHOENIX)") and the
# "w/ Radeon …" tail some CPUs carry.
_NOISE = re.compile(r"\(.*?\)|\bw/.*$")
_DROP_TOKENS = {"NVIDIA", "AMD", "Intel", "GeForce", "Graphics", "Processor", "CPU", "PRO"}


def _shorten(name: str) -> str:
    """ "NVIDIA GeForce RTX 5080" → "RTX 5080"; "AMD Ryzen 5 PRO 230 w/ Radeon
    760M Graphics" → "Ryzen 5 230". Keeps anything it doesn't recognise."""
    toks = [t for t in _NOISE.sub("", name).split()
            if t not in _DROP_TOKENS and not re.fullmatch(r"\d+-Core", t)]
    return " ".join(toks)


def _silicon(df: pd.DataFrame) -> dict[str, tuple[str, str]]:
    """Submission name → (cpu, gpu) short hardware names; gpu is "" when none is
    identifiable. Some machine blocks list no GPUs even though an iGPU ran —
    then the accelerated providers' `device` strings stand in (skipping wrappers
    like "webgpu", which shorten to a digit-less provider name, not hardware)."""
    out: dict[str, tuple[str, str]] = {}
    for m, sub in df.groupby("machine"):
        cpu = _shorten(sub.cpu.iloc[0])
        gpu = next((_shorten(g) for g in sub.gpu.unique() if g and g != "cpu"), "")
        if not gpu and "device" in sub:
            devices = sub.loc[sub.provider != "cpu", "device"].dropna().unique()
            gpu = next((s for s in map(_shorten, devices)
                        if s != cpu and any(c.isdigit() for c in s)), "")
        out[m] = (cpu, gpu)
    return out


def machine_labels(df: pd.DataFrame) -> dict[str, str]:
    """Submission name → hardware-descriptive display label.

    `monsieurtapir-laptop` says nothing to a reader; `Ryzen 5 230 + Radeon 760M`
    does. Built from each submission's own cpu/gpu strings (via `_silicon`); a
    machine whose GPU is its CPU (unified, or cpu-only) shows just the chip.
    Duplicate labels get the submission name appended so they stay distinct.
    """
    labels = {m: (cpu if gpu in ("", cpu) else f"{cpu} + {gpu}")
              for m, (cpu, gpu) in _silicon(df).items()}
    dupes = pd.Series(labels).duplicated(keep=False)
    return {m: f"{label} ({m})" if dupes[m] else label for m, label in labels.items()}


def lane_labels(df: pd.DataFrame) -> dict[str, dict[str, str]]:
    """Submission name → {"cpu": lane, "gpu": lane}.

    A *lane* is one piece of silicon a config can run on: a machine's CPU or its
    GPU — `Ryzen 9 9950X · cpu`, `RTX 5080 · gpu`. Named after the silicon, not
    the machine, so a matchup within a lane compares backends on identical
    hardware. A unified/unidentified GPU borrows the CPU's name (the `· gpu` tag
    still disambiguates). Lanes that collide across machines get the submission
    name appended.
    """
    lanes = {m: {"cpu": f"{cpu} · cpu", "gpu": f"{gpu or cpu} · gpu"}
             for m, (cpu, gpu) in _silicon(df).items()}
    for kind in ("cpu", "gpu"):
        vals = pd.Series({m: d[kind] for m, d in lanes.items()})
        for m in vals.index[vals.duplicated(keep=False)]:
            lanes[m][kind] += f" ({m})"
    return lanes


def prepare(
    df: pd.DataFrame, ladder: list[str] = TASK_LADDER, *, describe_machines: bool = True
) -> tuple[pd.DataFrame, list[str]]:
    """Order tasks along the context ladder, relabel machines, label configs
    and lanes.

    Returns (df, task_order). With `describe_machines` (default), `machine`
    becomes the hardware label from `machine_labels` and the original submission
    name moves to a `submission` column — narrative charts read hardware,
    the coverage appendix keeps the traceable name. Every row also gets its
    `lane` (see `lane_labels`): the silicon it ran on, cpu provider → the CPU
    lane, anything else → the GPU lane.

    A "config" is one (machine, backend, provider, quant) way of running a
    model. Machine and quant only enter the label when the frame holds more
    than one — a single-machine run reads as plain `ggml-cuda`, a cross-machine
    run disambiguates with `Ryzen 5 230 + Radeon 760M · ggml-vulkan`. Labels are
    computed on every row (ok or failed) so failed configs plot too.
    """
    df = df.copy()
    order = list(ladder)
    if df.empty:
        df["config"] = pd.Series(dtype=str)
        df["lane"] = pd.Series(dtype=str)
        return df, order

    order = [t for t in ladder if t in set(df.task)] + sorted(
        t for t in df.task.dropna().unique() if t not in ladder
    )
    df["task"] = pd.Categorical(df.task, categories=order, ordered=True)

    df["submission"] = df.machine
    _lanes = lane_labels(df)
    df["lane"] = [_lanes[m]["cpu" if p == "cpu" else "gpu"]
                  for m, p in zip(df.machine, df.provider, strict=True)]
    if describe_machines:
        df["machine"] = df.machine.map(machine_labels(df))

    multi_machine = df.machine.nunique() > 1
    multi_quant = df.quant.nunique() > 1

    def config(r):
        s = f"{r.backend}-{r.provider}"
        if multi_quant:
            s += f" {r.quant}"
        return f"{r.machine} · {s}" if multi_machine else s

    df["config"] = df.apply(config, axis=1)
    return df, order


def _melt_phases(g: pd.DataFrame, mapping: dict[str, str]) -> pd.DataFrame:
    """Wide stat columns → long (model, config, phase, value). NaN phases are
    dropped (e.g. VRAM on a CPU EP) so a config simply lacks that segment,
    rather than showing a zero."""
    out = g.melt(id_vars=["model", "config"], value_vars=list(mapping),
                 var_name="phase", value_name="value")
    out["phase"] = out.phase.map(mapping)
    return out.dropna(subset=["value"])


def time_phases(ok: pd.DataFrame) -> pd.DataFrame:
    """The cold-start-to-answer split: one row per (model, config, phase), ms.

    prefill = TTFT; decode = the rest of the turn (completion − TTFT). `ok`
    should already be sliced to one task; duplicate configs aggregate with max,
    erring pessimistic.
    """
    g = (ok.groupby(["model", "config"], observed=True)
         .agg(model_load=("model_load_ms_p50", "max"),
              context_init=("context_init_ms_p50", "max"),
              warmup=("warmup_ms_max", "max"),
              prefill=("ttft_ms_p50", "max"),
              completion=("completion_ms_p50", "max"))
         .reset_index())
    g["decode"] = g.completion - g.prefill
    return _melt_phases(g, {"model_load": "model load", "context_init": "context init",
                            "warmup": "warmup", "prefill": "prefill", "decode": "decode"})


def memory_phases(ok: pd.DataFrame) -> pd.DataFrame:
    """The decode-window footprint: one row per (model, config, phase), MB.

    RAM/VRAM = the sustained working set (median-across-spawns of the per-spawn
    decode median); "transient peak" tops the stack up to the decode high-water
    mark, so the stacked total reads as the peak the device must fit.
    """
    g = (ok.groupby(["model", "config"], observed=True)
         .agg(ram=("decode_rss_sustained_mb_p50", "max"),
              vram=("decode_vram_sustained_mb_p50", "max"),
              ram_peak=("decode_rss_peak_mb_max", "max"),
              vram_peak=("decode_vram_peak_mb_max", "max"))
         .reset_index())
    g["peak"] = (g.ram_peak.fillna(0) + g.vram_peak.fillna(0)
                 - g.ram.fillna(0) - g.vram.fillna(0)).clip(lower=0)
    return _melt_phases(g, {"ram": "RAM", "vram": "VRAM", "peak": "transient peak"})


def shared_config_order(frames: list[pd.DataFrame]) -> list[str]:
    """One config ordering for a set of per-task phase frames (from
    `time_phases` / `memory_phases`), so tabbed charts keep every config on the
    same row across tabs instead of re-sorting per task.

    Within each frame a config scores by its fastest/lightest model total (the
    same key `charts.stacked` sorts by); ranks then average across frames, a
    frame where the config has no sample ranking it last.
    """
    ranks = []
    for f in frames:
        tot = f.groupby(["model", "config"], observed=True)["value"].sum()
        ranks.append(tot.groupby("config").min().rank())
    r = pd.concat(ranks, axis=1)
    return r.fillna(len(r) + 1).mean(axis=1).sort_values().index.tolist()


def failures(df: pd.DataFrame) -> pd.DataFrame:
    """Configs with no usable sample in this slice: (model, config, label) rows,
    labelled "too slow" (timed out / below the floor) or "errored" (a spawn died
    producing nothing — crash/OOM, not slowness)."""
    f = df[df.status.isin(FAIL_LABELS)]
    out = f[["model", "config", "status"]].drop_duplicates().copy()
    out["label"] = out.status.map(FAIL_LABELS)
    return out[["model", "config", "label"]]


def status_cells(df: pd.DataFrame) -> pd.DataFrame:
    """Outcome counts per backend·provider: (backend, provider, who, status, n).

    A cell is one (machine, provider, model, task) attempt. An unhealthy config
    counts once — it failed its brain-check, so its tasks never ran.
    """
    cells = df[df.status.isin(["ok", "too_slow", "errored", "unhealthy"])].copy()
    cells["who"] = cells.backend + " · " + cells.provider
    return (cells.groupby(["backend", "provider", "who", "status"], observed=True)
            .size().reset_index(name="n"))


def _matchup(t: pd.DataFrame, scope: str, col: str) -> pd.DataFrame:
    """Per (scope, backend, model, task) keep the row minimizing `col`, then add
    the head-to-head pair view within each (scope, model, task) cell: lo/hi span
    the backends' values, ratio = hi/lo, and n_backends flags cells where only
    one backend produced a sample (so a ×1.0 "gap" isn't a tie, it's a
    walkover)."""
    best = t.loc[t.groupby([scope, "backend", "model", "task"], observed=True)[col]
                 .idxmin()].copy()
    grp = best.groupby([scope, "model", "task"], observed=True)
    best["lo"] = grp[col].transform("min")
    best["hi"] = grp[col].transform("max")
    best["n_backends"] = grp.backend.transform("nunique")
    best["ratio"] = best.hi / best.lo
    return best


def _with_total_s(ok: pd.DataFrame) -> pd.DataFrame:
    """Total wall clock from a cold process to a finished answer (load + context
    + warmup + completion), in seconds as `total_s`."""
    return ok.assign(total_s=(ok.model_load_ms_p50 + ok.context_init_ms_p50
                              + ok.warmup_ms_max + ok.completion_ms_p50) / 1000)


def best_of_backend(ok: pd.DataFrame) -> pd.DataFrame:
    """Each backend's fastest config per (machine, model, task) by `total_s`,
    with the head-to-head columns from `_matchup`."""
    return _matchup(_with_total_s(ok), "machine", "total_s")


def lane_time(ok: pd.DataFrame) -> pd.DataFrame:
    """Each backend's fastest config per (lane, model, task) by `total_s`, with
    the head-to-head columns from `_matchup`. Within a lane both backends ran on
    the same silicon, so the gap is the backend, not the hardware."""
    return _matchup(_with_total_s(ok), "lane", "total_s")


def lane_memory(ok: pd.DataFrame) -> pd.DataFrame:
    """Each backend's lightest config per (lane, model, task) by `peak_gb` —
    the high-water RAM+VRAM footprint across prefill *and* decode, i.e. what the
    device must actually fit (prefill can transiently dwarf decode: compile
    spikes, arena growth). Head-to-head columns from `_matchup`."""
    prefill = ok.prefill_rss_peak_mb_max.fillna(0) + ok.prefill_vram_peak_mb_max.fillna(0)
    decode = ok.decode_rss_peak_mb_max.fillna(0) + ok.decode_vram_peak_mb_max.fillna(0)
    t = ok.assign(peak_gb=pd.concat([prefill, decode], axis=1).max(axis=1) / 1024)
    return _matchup(t, "lane", "peak_gb")


def fallback_cost(ok: pd.DataFrame, backend: str = "ggml") -> pd.DataFrame:
    """The CPU-fallback tax as dumbbell pairs: a long view of `gpu_vs_cpu`,
    one row per (machine, model, task, phase, side) with absolute `seconds` —
    phase "TTFT" (prefill) or "decode" (the rest of the turn), side cpu/gpu.

    `leg` ("model · phase") keys one dumbbell row; lo/hi/ratio/n_backends
    mirror `_matchup` so `charts.dumbbell` can draw it (the gap label reads
    ×prefill_x on TTFT legs and ×decode-time on decode legs).
    """
    gvc = gpu_vs_cpu(ok, backend)
    parts = []
    for side in ("cpu", "gpu"):
        base = pd.DataFrame({
            "machine": gvc.machine, "model": gvc.model, "task": gvc.task,
            "provider": gvc[f"provider_{side}"], "side": side,
        })
        ttft = gvc[f"ttft_ms_p50_{side}"]
        parts.append(base.assign(phase="TTFT", seconds=ttft / 1000))
        parts.append(base.assign(
            phase="decode",
            seconds=(gvc[f"completion_ms_p50_{side}"] - ttft) / 1000))
    out = pd.concat(parts, ignore_index=True)
    out["leg"] = out.model.astype(str) + " · " + out.phase
    grp = out.groupby(["machine", "leg", "task"], observed=True)
    out["lo"] = grp.seconds.transform("min")
    out["hi"] = grp.seconds.transform("max")
    out["ratio"] = out.hi / out.lo
    out["n_backends"] = grp.side.transform("nunique")
    return out


def gpu_vs_cpu(ok: pd.DataFrame, backend: str = "ggml") -> pd.DataFrame:
    """Pair each CPU cell with the fastest accelerated provider for the same
    (machine, model, quant, task) within one backend.

    One row per pair, both sides' columns suffixed `_cpu`/`_gpu`, plus the
    speedups: `prefill_x` (TTFT ratio — compute-bound, where the GPU shines),
    `decode_x` (tok/s ratio — bandwidth-bound, where the gap narrows), and
    `completion_x` (whole-turn ratio — both phases weighted by where the time
    actually goes). Machines with no accelerated provider drop out (inner join).
    """
    g = ok[ok.backend == backend]
    keys = ["machine", "model", "quant", "task"]
    cpu = g[g.provider == "cpu"]
    acc = g[g.provider != "cpu"]
    if len(acc):
        acc = acc.loc[acc.groupby(keys, observed=True).decode_tps_p50.idxmax()]
    out = cpu.merge(acc, on=keys, suffixes=("_cpu", "_gpu"))
    out["prefill_x"] = out.ttft_ms_p50_cpu / out.ttft_ms_p50_gpu
    out["decode_x"] = out.decode_tps_p50_gpu / out.decode_tps_p50_cpu
    out["completion_x"] = out.completion_ms_p50_cpu / out.completion_ms_p50_gpu
    out["ttft_s_cpu"] = out.ttft_ms_p50_cpu / 1000
    out["ttft_s_gpu"] = out.ttft_ms_p50_gpu / 1000
    return out
