"""Derived frames for the comparison notebook.

`load` builds the raw tidy frame; this module derives the views the notebook
plots. Pure pandas, no plotting, unit-tested — so the notebook stays a
disposable arrangement of tested pieces.
"""

from __future__ import annotations

import re

import pandas as pd

# The validation job; unknown tasks (a renamed job) append after it.
TASK_LADDER = ["summarize-large"]

# Stacked-chart segment labels, bottom→top draw order. `charts` keys its
# palettes off these, so the labels live in exactly one place.
TIME_PHASES = ["model load", "context init", "warmup", "prefill", "decode"]
MEMORY_PHASES = ["weights", "KV cache", "compute"]

FAIL_LABELS = {"too_slow": "too slow", "errored": "errored"}


# Vendor noise in cpu/gpu strings: parentheticals ("(RADV PHOENIX)") and the
# "w/ Radeon …" tail some CPUs carry.
_NOISE = re.compile(r"\(.*?\)|\bw/.*$")
_DROP_TOKENS = {"NVIDIA", "AMD", "Intel", "GeForce", "Graphics", "Processor", "CPU", "PRO"}


def _shorten(name: str) -> str:
    """ "NVIDIA GeForce RTX 5080" → "RTX 5080"; "AMD Ryzen 5 PRO 230 w/ Radeon
    760M Graphics" → "Ryzen 5 230". Keeps anything it doesn't recognise; a name
    that is *all* dropped tokens ("Intel(R) Graphics (MTL)") keeps its de-noised
    form ("Intel Graphics") rather than vanishing."""
    cleaned = _NOISE.sub("", name).split()
    toks = [t for t in cleaned
            if t not in _DROP_TOKENS and not re.fullmatch(r"\d+-Core", t)]
    return " ".join(toks or cleaned)


def _silicon(df: pd.DataFrame) -> dict[str, tuple[str, str]]:
    """Submission name → (cpu, gpu) short hardware names; gpu is "" when none is
    identifiable. Some machine blocks list no GPUs even though an iGPU ran —
    then the iGPU named in the CPU string ("… w/ Radeon 780M Graphics") stands
    in, else the accelerated providers' `device` strings (skipping wrappers like
    "webgpu", which shorten to a provider name, not hardware). A name with no
    model number ("Intel Graphics") identifies nothing and is discarded — the
    lane borrows the CPU's name instead, which at least names the die."""
    out: dict[str, tuple[str, str]] = {}
    for m, sub in df.groupby("machine"):
        raw_cpu = sub.cpu.iloc[0]
        cpu = _shorten(raw_cpu)
        gpu = next((_shorten(g) for g in sub.gpu.unique() if g and g != "cpu"), "")
        if not gpu and (tail := re.search(r"\bw/\s*(.+)$", raw_cpu)):
            gpu = _shorten(tail.group(1))
        if not gpu and "device" in sub:
            eps = {p.lower() for p in sub.provider.dropna()}
            devices = sub.loc[sub.provider != "cpu", "device"].dropna().unique()
            gpu = next((s for s in map(_shorten, devices)
                        if s and s != cpu and s.lower() not in eps), "")
        if not re.search(r"\d", gpu):
            gpu = ""
        out[m] = (cpu, gpu)
    return out


def machine_labels(df: pd.DataFrame) -> dict[str, str]:
    """Submission name → hardware-descriptive display label.

    A submission name says nothing to a reader; `Ryzen 5 230 (Radeon 760M)`
    does. Built from each submission's own cpu/gpu strings (via `_silicon`); a
    machine whose GPU is its CPU (unified, or cpu-only) shows just the chip.
    Duplicate labels get the submission name appended so they stay distinct.
    """
    labels = {m: (cpu if gpu in ("", cpu) else f"{cpu} ({gpu})")
              for m, (cpu, gpu) in _silicon(df).items()}
    dupes = pd.Series(labels).duplicated(keep=False)
    return {m: f"{label} ({m})" if dupes[m] else label for m, label in labels.items()}


def lane_labels(df: pd.DataFrame) -> dict[str, dict[str, str]]:
    """Submission name → {"cpu": lane, "gpu": lane}.

    A *lane* is one piece of silicon a config can run on: a machine's CPU or its
    GPU — `Ryzen 9 9950X · cpu`, `RTX 5080 · gpu`. Named after the silicon, not
    the machine, so rows within a lane ran on identical hardware. A
    unified/unidentified GPU borrows the CPU's name (the `· gpu` tag
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
    model, labeled by the silicon it ran on — `RTX 5080 · ggml-vulkan`, not the
    machine name. Quant enters only when the frame holds more than one; a
    single-machine run reads as plain `ggml-cuda`. Identical labels from
    different submissions get the submission name appended. Labels are computed
    on every row (ok or failed) so failed configs plot too.
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

    multi_machine = df.submission.nunique() > 1
    multi_quant = df.quant.nunique() > 1
    _chips = _silicon(df.assign(machine=df.submission))

    def config(r):
        s = f"{r.backend}-{r.provider}"
        if multi_quant:
            s += f" {r.quant}"
        if not multi_machine:
            return s
        cpu, gpu = _chips[r.submission]
        chip = cpu if r.provider == "cpu" else (gpu or cpu)
        return f"{chip} · {s}"

    df["config"] = df.apply(config, axis=1)
    # The same silicon on two submissions makes identical labels — append the
    # submission name so configs stay distinct rows, not silently pooled ones.
    span = df.groupby("config").submission.transform("nunique")
    df.loc[span > 1, "config"] = df.config + " (" + df.submission + ")"
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


def memory_model(mem: pd.DataFrame, ok: pd.DataFrame, at: int = 2048
                 ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Predicted-vs-measured memory at the job's operating point.

    `mem` is the memory-cost-curve frame (`load_memory`) and `ok` the scored-job
    slice, each already carrying a shared `config` label. Bars (long: model,
    config, phase, value MB): the allocator's point at n_ctx == `at` — the
    job's context — split into weights / KV cache / compute workspace, pooled
    over host and device buffers. Ticks (model, config, value): what the job
    actually occupied — the sustained decode footprint from the outside
    sampler, RSS and VRAM pooled. Configs without a curve point get no bar
    (never a zero); configs without a scored job get no tick.
    """
    if len(mem):
        g = (mem[mem.n_ctx == at].groupby(["model", "config"], observed=True)
             .agg(weights=("weights_mb", "max"), kv=("kv_mb", "max"),
                  compute=("compute_mb", "max")).reset_index())
        bars = _melt_phases(g, {"weights": "weights", "kv": "KV cache",
                                "compute": "compute"})
    else:
        bars = pd.DataFrame(columns=["model", "config", "phase", "value"])
    t = (ok.groupby(["model", "config"], observed=True)
         .agg(rss=("decode_rss_sustained_mb_p50", "max"),
              vram=("decode_vram_sustained_mb_p50", "max"))
         .reset_index())
    ticks = t.dropna(subset=["rss"]).copy()
    ticks["value"] = ticks.rss + ticks.vram.fillna(0)
    return bars, ticks[["model", "config", "value"]]


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
