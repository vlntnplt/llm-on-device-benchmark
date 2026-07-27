"""Cross-machine benchmark — exploration & comparison.

A marimo notebook (a plain .py file — no jupytext pairing, diffs cleanly). The
tested package does the work: `bench_analysis.load` builds the tidy frames,
`bench_analysis.prep` derives the views, `bench_analysis.charts` owns the visual
language. This notebook just arranges them and writes the prose — it is the
disposable part.

    uv run --project analysis marimo edit analysis/report.py            # interactive
    uv run --project analysis marimo export html analysis/report.py -o report.html

Export to HTML to hand the team a static snapshot; add `--no-include-code` so it
reads as a report rather than a notebook. The export has no kernel, so nothing
here may depend on a reactive widget — every switch in the report is a
`bench_analysis.switcher` group: panels pre-rendered, plain radios and CSS doing
the switching.

Four sections, one per kind of measurement, every number computed live from
the published submissions:

    1. the machines — hardware, installed memory, probed device ceilings
    2. the measured cost curves — prefill vs prompt length, decode vs KV fill
    3. the validation job — end-to-end time, memory, and reliability
    4. GPU vs CPU on the same silicon — what a CPU fallback costs
"""

import marimo

__generated_with = "0.23.9"
app = marimo.App(
    width="medium",
    app_title="On-device LLM inference benchmark",
    css_file="report.css",
)


@app.cell
def _():
    from pathlib import Path

    import marimo as mo
    import pandas as pd

    from bench_analysis import (
        charts,
        load_memory,
        load_probes,
        load_results,
        load_sweeps,
        prep,
        switcher,
    )

    return (Path, charts, load_memory, load_probes, load_results, load_sweeps,
            mo, pd, prep, switcher)


@app.cell
def _(mo):
    mo.md("""
    # Cross-machine on-device LLM benchmark

    One inference stack — **ggml**
    ([llama.cpp](https://github.com/ggml-org/llama.cpp), a native C++ stack
    consuming GGUF files) — measured on each contributor's machine. Every
    measurement launches a fresh process, loads at most one model on one
    compute provider (`cpu`, `vulkan`, `metal`, …), does one unit of work, and
    exits. A **config** is one way of running a model: a machine, a provider,
    and a quantization.

    A submission measures a machine's **cost function**, and the report reads
    as that argument, in order:

    1. what the silicon can do bare — per-provider **device ceilings**, before
       any model is loaded;
    2. what inference actually achieves against those ceilings as the work
       grows — the **cost curves**, swept with synthetic tokens to 8k context;
    3. whether the curves predict reality — one **real job** (a ~1700-token
       document summarized on a 256-token budget, decoded **greedily, to a
       fixed count, EOS ignored**), measured end to end and plotted onto the
       curves it should land on;
    4. what that all costs when the GPU isn't available — the **CPU
       fallback**, from the same measurements.

    Before any of that, a config must answer three trivial questions correctly
    (the **brain-check**) or it is reported `unhealthy` rather than timed.
    Prompts render through each model's own chat template with reasoning off.

    Two reading notes. A quantization *label* names stack-specific math
    (`q4` is Q4_K_M), so comparisons hold *within* a label, qualified by the
    stack versions each submission records. And every number on this page —
    including the ones in the text — is computed from the published submissions
    in `results/published/`, so the prose updates as new runs land.
    """)
    return


@app.cell
def _(Path, load_memory, load_probes, load_results, load_sweeps, mo, pd, prep):
    # Anchor to the notebook's own location (…/analysis) so the path holds no matter
    # the cwd `marimo edit` was launched from; results live one dir up.
    _nb = mo.notebook_dir()
    _root = (_nb.parent / "results") if _nb else Path("../results")

    # The published submissions are the shared baseline: every PR-merged run under
    # results/published/<name>/.
    _published = _root / "published"
    _df = load_results(_published)
    sweeps = load_sweeps(_published)
    probes = load_probes(_published)
    memory = load_memory(_published)

    # Preview a not-yet-published run alongside the baseline: point PREVIEW at a
    # local results dir (what `bench run --out` wrote — e.g. _root / "my-box") to
    # fold it in before you `bench publish` it. It loads under its in-file host name
    # with a " · preview" tag so it's distinct from the published rows. None = off.
    PREVIEW = None
    if PREVIEW is not None:
        for _loader, _target in [(load_results, "_df"), (load_sweeps, "sweeps"),
                                 (load_probes, "probes"), (load_memory, "memory")]:
            _prev = _loader(PREVIEW)
            if len(_prev):
                _prev["machine"] = _prev["machine"].astype(str) + " · preview"
            if _target == "_df":
                _df = pd.concat([_df, _prev], ignore_index=True)
            elif _target == "sweeps":
                sweeps = pd.concat([sweeps, _prev], ignore_index=True)
            elif _target == "probes":
                probes = pd.concat([probes, _prev], ignore_index=True)
            else:
                memory = pd.concat([memory, _prev], ignore_index=True)

    df, task_order = prep.prepare(_df)
    ok = df[df.status == "ok"].copy()
    return df, memory, ok, probes, sweeps, task_order


@app.cell
def _(df, mo, probes):
    # One row per machine: what it is (incl. installed memory — the source of
    # its nominal bandwidth) — and below, per provider, what its silicon can do
    # bare. The ceilings are the denominators for every number that follows.
    _specs = df.drop_duplicates("submission")
    _spec_rows = []
    for _r in _specs.itertuples():
        _ram = f"{_r.ram_gb:g} GB" if _r.ram_gb == _r.ram_gb else "?"
        if _r.ram_channels == _r.ram_channels and _r.ram_channels:
            _ram += f", {int(_r.ram_channels)}-ch @ {int(_r.ram_mts)} MT/s"
        _spec_rows.append(f"| {_r.machine} | {_r.cpu} | {_r.gpu} | {_ram} |")

    _ceil_rows = []
    for (_m, _p), _g in probes[probes.status == "ok"].groupby(["machine", "provider"]):
        _gemm = _g[_g.kind == "gemm"].tflops.max()
        _copies = {r.kind: r.gbs for r in _g[_g.kind != "gemm"].itertuples()}
        _ceil_rows.append(
            f"| {_m} | {_p} | {_gemm:.1f} | {_copies.get('d2d', float('nan')):.0f} "
            f"| {_copies.get('h2d', float('nan')):.1f} |")

    mo.md(f"""
    ## 1 · The machines

    | machine | CPU | GPU | memory |
    |---|---|---|---|
    {chr(10).join(_spec_rows)}

    What each provider's silicon does **bare** — an f16 matrix multiply shaped
    like a prefill micro-batch, and buffer copies — before any model touches
    it. Every model number below can be read against these ceilings, so a slow
    curve is attributable to the runtime or to the device, not guessed at.
    `d2d` counts read+write traffic (the STREAM convention); on CPU and
    unified memory `h2d` is the same memory, so a gap between the two columns
    is the PCIe link of a discrete card.

    | machine | provider | gemm TFLOP/s | d2d GB/s | h2d GB/s |
    |---|---|---|---|---|
    {chr(10).join(_ceil_rows)}
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ## 2 · The measured cost curves

    The sweep measures the cost function in **one instrumented pass**, with
    synthetic tokens and no chat semantics: a full-context prefill timed
    ubatch-chunk by ubatch-chunk — each chunk's cost is the marginal cost at
    its depth, so the attention term is measured directly as the series'
    slope. **Prefill** below plots the cumulative time: on a log-log axis a
    purely compute-bound prefill is a straight line of slope 1, and any
    upward bend is the attention term arriving. The pass leaves the cache
    primed, so **decode** rate is measured at the reached depth and then at
    trimmed fills below it — the downward drift is the per-token cost of
    reading an ever-larger cache, measured rather than modelled.

    **Memory** is swept the same way: the allocator's KV/state reservation,
    re-measured at a ladder of context sizes — exact numbers, no repeats
    needed. A line through the origin is plain per-token KV growth; a curve
    that flattens at short context is a recurrent or hybrid architecture's
    constant state showing itself. (Weights and compute workspace barely move
    with context; they appear in §3's memory breakdown instead.)

    Curves stop early where the sweep budget ended: on slow silicon the
    measured envelope shrinks instead of the time growing, so a short curve
    is a small envelope, not missing data.

    The **diamonds** are not sweep points: they are §3's validation job — a
    real chat-templated workload, measured end to end — placed at its
    ≈1.7k-token prompt. A diamond on its curve is the sweep keeping its
    promise; a diamond off it is the gap between synthetic and real work,
    shown rather than assumed away.
    """)
    return


@app.cell
def _(charts, memory, mo, ok, pd, sweeps, switcher):
    # The job's ≈prompt length (tokens are backend-tokenized; the task doc is
    # authored to ~1700). Its decode runs at fills 1700→1956; the overlay sits
    # at the midpoint.
    JOB_TOKENS = 1700

    _s = sweeps.copy()
    _s["config"] = _s.machine + " · " + _s.provider
    _mem = (memory.assign(config=memory.machine + " · " + memory.provider)
            if len(memory) else memory)
    _job = ok.assign(config=ok.submission + " · " + ok.provider)
    _pre_overlay = pd.DataFrame({
        "config": _job.config, "model": _job.model,
        "x": JOB_TOKENS, "y": _job.ttft_ms_p50}).dropna()
    _dec_overlay = pd.DataFrame({
        "config": _job.config, "model": _job.model,
        "x": JOB_TOKENS + 128, "y": _job.decode_tps_p50}).dropna()

    # One tab per model, both curves in the tab; lines = machine · provider.
    def _curve_panel(model):
        rows = _s[_s.model == model]
        over = _pre_overlay[_pre_overlay.model == model]
        html = mo.as_html(charts.curves(
            rows[rows.kind == "prefill"],
            "prefill: cumulative wall time vs prompt depth (log-log), one "
            "instrumented pass — ◆ = the validation job's TTFT",
            x="tokens", y="ttft_ms",
            x_title="prompt tokens", y_title="ms",
            overlay=over if len(over) else None)).text
        dec = rows[rows.kind == "decode"]
        dover = _dec_overlay[_dec_overlay.model == model]
        if len(dec):
            html += mo.as_html(charts.curves(
                dec.assign(kv_fill=lambda d: d.kv_fill.clip(lower=64)),
                "decode: tok/s vs KV-cache fill (log x; fill 0 shown at 64) — "
                "◆ = the validation job",
                x="kv_fill", y="tps_p50", lo="tps_min", hi="tps_max",
                x_title="KV cache fill (tokens)", y_title="tok/s", log_y=False,
                overlay=dover if len(dover) else None)).text
        memc = _mem[_mem.model == model] if len(_mem) else _mem
        if len(memc):
            html += mo.as_html(charts.curves(
                memc, "memory: KV/state reservation vs context (log-log) — "
                "exact allocator numbers",
                x="n_ctx", y="kv_mb",
                x_title="context (tokens)", y_title="MB")).text
        return html

    _models = sorted(_s.model.dropna().unique())
    mo.Html(switcher.tabs({m: _curve_panel(m) for m in _models}, group="curves")
            if _models else "")
    return (JOB_TOKENS,)


@app.cell
def _(JOB_TOKENS, df, mo, ok, prep, sweeps, task_order):
    import math as _math

    # The reliability verdict is one sentence when everything scored; failures
    # get named only when they exist.
    _counts = prep.status_cells(df)
    _bad_cells = _counts[_counts.status != "ok"]
    _ok_n, _all_n = int(_counts[_counts.status == "ok"].n.sum()), int(_counts.n.sum())
    _n_machines = df.machine.nunique()
    _mach = f"{_n_machines} machine" + ("s" if _n_machines != 1 else "")

    _verdict = f"**{_ok_n}/{_all_n} jobs scored** across {_mach}."
    if len(_bad_cells):
        _verdict += " The misses: " + "; ".join(
            f"{r.n}× {r.who} {r.status.replace('_', ' ')}" for r in _bad_cells.itertuples()) + "."
    _bad_sweeps = df[df.sweep_status.isin(["too_slow", "errored"])]
    if len(_bad_sweeps):
        _verdict += " Sweeps that did not complete: " + "; ".join(
            f"{r.machine} · {r.provider} · {r.model} ({r.sweep_status.replace('_', ' ')})"
            for r in _bad_sweeps.itertuples()) + "."

    # How far each job TTFT sits from its own prefill curve, log-interpolated
    # at the job's prompt length — the report's honesty figure, computed live.
    def _interp(points, at):
        pts = sorted(points)
        for (x0, y0), (x1, y1) in zip(pts, pts[1:], strict=False):
            if x0 <= at <= x1:
                f = (_math.log(at) - _math.log(x0)) / (_math.log(x1) - _math.log(x0))
                return _math.exp(_math.log(y0) + f * (_math.log(y1) - _math.log(y0)))
        return None

    _deltas = []
    for _r in ok.itertuples():
        _cs = sweeps[(sweeps.machine == _r.submission) & (sweeps.provider == _r.provider)
                     & (sweeps.model == _r.model) & (sweeps.kind == "prefill")]
        _pred = _interp(list(zip(_cs.tokens, _cs.ttft_ms, strict=True)), JOB_TOKENS)
        if _pred and _r.ttft_ms_p50 == _r.ttft_ms_p50:
            _deltas.append(abs(_r.ttft_ms_p50 / _pred - 1) * 100)
    _agree = (f" Interpolating each config's prefill curve at the job's prompt "
              f"length puts the measured TTFTs a median "
              f"**{sorted(_deltas)[len(_deltas) // 2]:.0f}%** "
              f"from the curve (worst {max(_deltas):.0f}%)."
              if _deltas else "")

    mo.md(f"""
    ## 3 · The validation job

    One real workload — `{task_order[0] if task_order else "—"}` — run end to
    end on every config: the check that the curves above predict an actual
    job, and the only place cold-start, load, and memory are measured.
    {_verdict}{_agree}

    Each bar is one config; its length is the full wall-clock time from a cold
    process to a finished answer, split into the five phases every run goes
    through: **model load** (weights to device), **context init** (KV cache +
    graph), **warmup** (one-time kernel/JIT pass), **prefill** (ingest the
    prompt — this segment is the TTFT), **decode** (generate). Pale = setup
    paid once per process; vivid = the generation work paid on every request.
    The fastest config per model gets a green total; configs with no usable
    sample stay visible as full-width markers.
    """)
    return


@app.cell
def _(charts, df, mo, ok, prep, switcher):
    # One tab per model; a shared config order keeps every config on the same
    # row across tabs.
    _models = sorted(df.model.dropna().unique())
    _frames = {_m: prep.time_phases(ok[ok.model == _m]) for _m in _models}
    _nonempty = [f for f in _frames.values() if len(f)]
    _order = prep.shared_config_order(_nonempty) if _nonempty else None
    _panels = {
        _m: mo.as_html(charts.stacked(
            _frames[_m], charts.TIME_COLORS,
            "time to a finished answer (ms) — load + prefill + decode, lower is better",
            dnf=prep.failures(df[df.model == _m]), config_order=_order)).text
        for _m in _models
    }
    mo.Html(switcher.tabs(_panels, group="time") if _panels else "")
    return


@app.cell
def _(mo):
    mo.md("""
    The memory the job needs, said two ways. The **bars** are computed from
    the allocator's own reservations, reported by the runtime and pooled over
    host and device buffers: **weights**, the **KV cache**, and the
    **compute** workspace. Each sweep measures that breakdown at a ladder of
    context sizes — the memory cost curve rides in every submission for
    fitting — and the bars show the point at the job's context (2048). The
    **tick** is measured from outside: the sustained RSS+VRAM the sampler saw
    while the job decoded. The distance between a tick and its bar end is the
    part of the footprint the allocator doesn't own (runtime, tokenizer,
    host-side scaffolding) — shown, not modelled away.
    """)
    return


@app.cell
def _(charts, memory, mo, ok, prep):
    # Config labels match the curves chart (machine · provider); the bars come
    # from the sweep's memory curve, the ticks from the job's sampler.
    _mem = memory.assign(config=memory.machine + " · " + memory.provider) \
        if len(memory) else memory
    _job = ok.assign(config=ok.submission + " · " + ok.provider)
    _bars, _ticks = prep.memory_model(_mem, _job, at=2048)
    mo.Html(mo.as_html(charts.stacked(
        _bars, charts.MEMORY_COLORS,
        "memory at the job's context (MB): bars = allocator breakdown, "
        "tick = measured sustained footprint",
        ticks=_ticks)).text) if len(_bars) else mo.md(
        "*(these submissions carry no allocator memory curve)*")
    return


@app.cell
def _(mo, ok):
    # The harness keeps one decoded completion per job so a run stays
    # eyeball-inspectable — proof the timings came off a model that was
    # actually writing, not emitting filler at speed.
    import html as _html

    def _quote(row):
        _who = f"{row.machine} · {row.model} {row.quant} · {row.provider}"
        return (f'<blockquote><span class="who">{_html.escape(_who)}</span>'
                f'{_html.escape(str(row.sample_completion).strip())}</blockquote>')

    _with_text = ok[ok.sample_completion.notna()].groupby("model", observed=True).head(1)
    mo.Html('<details class="sample-completions">'
            '<summary>generated summaries, one per model</summary>'
            + "".join(_quote(r) for r in _with_text.itertuples()) + "</details>")
    return


@app.cell
def _(ok, prep):
    gvc = prep.gpu_vs_cpu(ok)

    # One verdict row per machine: speedups averaged over models. The phase
    # asymmetry headline is computed, not asserted — it holds whatever runs land.
    gvc_asym = (f"the GPU is **×{gvc.prefill_x.min():.1f}–{gvc.prefill_x.max():.0f}**"
                f" faster at prefill (compute-bound) but only "
                f"**×{gvc.decode_x.min():.1f}–{gvc.decode_x.max():.1f}** at decode "
                f"(bandwidth-bound)") if len(gvc) else ""

    _rows = []
    for _m, _sub in gvc.groupby("machine", observed=True):
        _rows.append(
            f"| {_m} | {_sub.provider_gpu.iloc[0]} "
            f"| ×{_sub.prefill_x.mean():.1f} "
            f"| ×{_sub.decode_x.mean():.1f} "
            f"| ×{_sub.completion_x.mean():.1f} |"
        )
    gvc_table = "\n".join(_rows)
    return gvc, gvc_asym, gvc_table


@app.cell
def _(gvc_asym, gvc_table, mo):
    mo.md(f"""
    ## 4 · GPU vs CPU on the same silicon

    On the machines that have both, what does running on the CPU instead of
    the GPU cost? The two generation phases stress different parts of the
    chip — prefill processes the whole prompt in parallel and is
    compute-bound; decode produces one token at a time and is bound by memory
    bandwidth — so the measured cost is lopsided: {gvc_asym}. A CPU fallback
    doesn't slow everything proportionally; the prompt-reading side moves
    most. One row per machine, speedups averaged over models; **full turn**
    is prefill + decode together, the two phases weighted by where the time
    actually goes.

    | machine | GPU | prefill speedup | decode speedup | full-turn speedup |
    |---|---|---|---|---|
    {gvc_table}
    """)
    return


if __name__ == "__main__":
    app.run()
