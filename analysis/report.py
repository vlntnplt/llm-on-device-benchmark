"""Cross-machine benchmark — the shared report.

A marimo notebook (a plain .py file — no jupytext pairing, diffs cleanly). The
tested package does the work: `bench_analysis.load` builds the tidy frames,
`bench_analysis.prep` derives the views, `bench_analysis.estimate` factors
costs from silicon, `bench_analysis.charts` owns the visual language. This
notebook arranges them and writes the prose — it is the disposable part.

    uv run --project analysis marimo edit analysis/report.py            # interactive
    uv run --project analysis marimo export html analysis/report.py -o report.html

The export has no kernel, so nothing here may depend on a reactive widget —
every switch is a `bench_analysis.switcher` group: panels pre-rendered, plain
radios and CSS doing the switching.

Three tabs:

    Models  — one performance card per model, and the numbers to compare them
    Fleet   — the estimator: model costs × machine ceilings, and how well that
              transfers (leave-one-out), toward fleet-reach predictions
    Explore — everything measured: machines, ceilings, cost curves, the
              validation job
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
        estimate,
        load_memory,
        load_probes,
        load_results,
        load_sweeps,
        prep,
        switcher,
    )

    return (Path, charts, estimate, load_memory, load_probes, load_results,
            load_sweeps, mo, pd, prep, switcher)


@app.cell
def _(mo):
    mo.md("""
    # On-device LLM inference benchmark

    What does it cost to run a small LLM on the machines people actually own?
    One inference stack — [llama.cpp](https://github.com/ggml-org/llama.cpp)
    — measured the same way on every contributed machine, on its GPU and on
    its CPU. Every number on this page is computed from the published runs in
    `results/published/`; nothing is quoted from spec sheets.

    **Models** compares the models measured. **Fleet** works toward the real
    question — *how much of a device fleet could run this model, acceptably?*
    **Explore** holds everything measured, chart by chart.
    """)
    return


@app.cell
def _(Path, load_memory, load_probes, load_results, load_sweeps, mo, prep):
    # Anchor to the notebook's own location (…/analysis) so the path holds no
    # matter the cwd `marimo edit` was launched from; results live one dir up.
    _nb = mo.notebook_dir()
    _root = (_nb.parent / "results") if _nb else Path("../results")
    _published = _root / "published"

    _df = load_results(_published)
    sweeps = load_sweeps(_published)
    probes = load_probes(_published)
    memory = load_memory(_published)

    df, task_order = prep.prepare(_df)
    ok = df[df.status == "ok"].copy()
    return df, memory, ok, probes, sweeps, task_order


@app.cell
def _(df, estimate, memory, mo, sweeps, switcher):
    # ── Models tab ──────────────────────────────────────────────────────────
    # The comparison table: machine-independent costs per model (what the model
    # *is*), then measured anchors per lane (what it *does* on known silicon).
    _costs = estimate.model_costs(df, memory)

    def _fits(row, gb):
        # Fits = weights + state + KV at 4k + ~1 GB runtime headroom.
        need = (row.file_bytes / 2**30 + row.kv_state_mb / 1024
                + row.kv_slope_mb * 4096 / 1024 + 1.0)
        return need <= gb

    _rows = []
    for _r in _costs.itertuples():
        _kv1k = _r.kv_slope_mb * 1024
        _rows.append(
            f"| **{_r.model}** {_r.quant} | {_r.file_bytes / 2**30:.1f} GB "
            f"| {_r.body_bytes / 2**30:.1f} GB "
            f"| {_kv1k:.0f} MB{f' + {_r.kv_state_mb:.0f} MB state' if _r.kv_state_mb >= 1 else ''} "
            f"| {'✓' if _fits(_r, 8) else '✗'} / {'✓' if _fits(_r, 16) else '✗'} |")

    _intro = mo.md(f"""
    Each model, reduced to the numbers that decide whether and how it runs.
    **On disk** is the download; **streamed per token** is the weight bytes a
    single decoded token actually reads — with the KV cache growth, it sets
    decode speed on any machine (decode is memory-bound: tokens/s ≈ effective
    memory bandwidth ÷ bytes per token). **KV per 1k tokens** is how the
    footprint grows with context; a "state" term is a recurrent architecture's
    constant. **Fits 8 / 16 GB** allows 4k context and ~1 GB of runtime
    overhead.

    | model | on disk | streamed per token | KV per 1k tokens | fits 8 / 16 GB |
    |---|---|---|---|---|
    {chr(10).join(_rows)}

    The cards below add measured rates on the machines we have, read off the
    sweep curves at standard reference points, and any caveats the runs
    surfaced.
    """).text

    # Per-model cards. Anchors come from the sweep — the fine measurement —
    # read at standard reference points, never from the single validation job
    # (that one exists to check the sweep, and lives in Explore).
    import math as _math

    def _interp_log(pts, at):
        pts = sorted(pts)
        if not pts or not (pts[0][0] <= at <= pts[-1][0]):
            return None
        for (x0, y0), (x1, y1) in zip(pts, pts[1:], strict=False):
            if x0 <= at <= x1:
                f = (_math.log(at / x0)) / (_math.log(x1 / x0))
                return y0 * (y1 / y0) ** f
        return None

    def _anchor_rows(model):
        rows = []
        for (m, p), g in sweeps[sweeps.model == model].groupby(["machine", "provider"]):
            dec = g[(g.kind == "decode") & g.tps_p50.gt(0)]
            pre = g[(g.kind == "prefill") & g.ttft_ms.notna()]
            if not len(dec) and not len(pre):
                continue
            fresh = dec.loc[dec.kv_fill.idxmin()].tps_p50 if len(dec) else None
            deep = dec.loc[dec.kv_fill.idxmax()] if len(dec) else None
            t2k = _interp_log(list(zip(pre.tokens, pre.ttft_ms, strict=True)), 2048) \
                if len(pre) else None
            t8k = _interp_log(list(zip(pre.tokens, pre.ttft_ms, strict=True)), 8192) \
                if len(pre) else None
            rows.append(
                f"| {m} · {p} "
                f"| {f'{t2k / 1e3:.1f} s' if t2k else '—'} "
                f"| {f'{t8k / 1e3:.1f} s' if t8k else '—'} "
                f"| {f'{fresh:.0f}' if fresh else '—'} "
                f"| {f'{deep.tps_p50:.0f} @ {deep.kv_fill / 1024:.0f}k' if deep is not None else '—'}"
                " |")
        return rows

    def _card(model):
        rows = _costs[_costs.model == model]
        anchors = _anchor_rows(model)
        bad = df[(df.model == model) & (df.status != "ok")]
        caveat = ""
        if len(bad):
            caveat = "\n**Did not score the job:** " + "; ".join(
                f"{b.lane} — {b.status.replace('_', ' ')}" for b in bad.itertuples()) + "\n"
        c = rows.iloc[0]
        return mo.md(f"""
        **Footprint** — {c.file_bytes / 2**30:.1f} GB on disk;
        {c.body_bytes / 2**30:.1f} GB streamed per decoded token;
        KV {c.kv_slope_mb * 1024:.0f} MB per 1k tokens of context{
        f" plus a {c.kv_state_mb:.0f} MB recurrent state" if c.kv_state_mb >= 1 else ""}.

        **Measured rates**, read off the sweep at reference points — how long
        to read a 2k / 8k prompt, and generation speed on an empty vs full
        context (generation slows as the context fills; both ends shown):

        | lane | prompt 2k | prompt 8k | tok/s fresh | tok/s deep |
        |---|---|---|---|---|
        {chr(10).join(anchors) if anchors else "| *(no sweep data)* | | | | |"}
        {caveat}
        The full curves these points are read from are in **Explore**; a "—"
        means the sweep's budget ended before that depth on that lane.
        """).text

    _models = sorted(_costs.model.unique())
    models_tab = _intro + switcher.tabs(
        {m: _card(m) for m in _models}, group="cards")
    return (models_tab,)


@app.cell
def _(df, estimate, memory, mo, probes, sweeps):
    # ── Fleet tab ───────────────────────────────────────────────────────────
    # The estimator's promise: model costs (Models tab) × a machine's ceilings
    # (memory GB, bandwidth GB/s, compute TFLOP/s) × a measured efficiency
    # factor → predicted rates for machines nobody measured. The tab reports
    # its own calibration honestly; the interactive fleet calculator lands
    # once the leave-one-out error earns it.
    _pts = estimate.points(df, sweeps, memory, probes)
    _fits = estimate.lane_fits(_pts)
    _loo = estimate.loo(_pts)

    _fit_rows = [
        f"| {r.machine} | {r.provider} | {r.kind} | {r.t0_ms:.1f} ms "
        f"| {'—' if not (r.rate < float('inf')) else f'{r.rate:.0f}'} "
        f"| {'—' if not (r.eta < float('inf')) else f'{r.eta:.2f}'} "
        f"| {r.r2:.2f} |"
        for r in _fits.itertuples()]
    _loo_rows = [
        f"| {r.machine} | {r.provider} | {r.kind} | {r.n_lanes_pooled} "
        f"| {r.median_err * 100:.0f}% | {r.worst_err * 100:.0f}% |"
        for r in _loo.itertuples()]

    fleet_tab = mo.md(f"""
    The goal: describe a fleet (so many iGPU laptops, so many discrete-GPU
    desktops, so many Macs; low/mid/high in each), describe a workload
    (prompt size, output size, context) and UX budgets (time to first token,
    tokens per second) — and read off, per model, **how much of the fleet
    clears the bar**.

    The mechanism: on every measured lane, time per unit of work follows
    `time = t₀ + work ÷ rate` — a per-token overhead plus a work term. The
    work is the model's cost (Models tab: bytes streamed per token for
    generation, FLOPs for prompt reading) — machine-independent. The lane
    contributes two numbers: its overhead `t₀` and its achieved `rate`,
    expressed as a fraction **η** of the machine's bare probed ceiling. If
    `t₀` and η transfer between machines of a class, a fleet only needs
    describable ceilings — memory, bandwidth, compute — not measurement.

    That model fits each measured lane well (generation R² ≥ 0.90 wherever
    the rate is identifiable):

    | machine | provider | phase | t₀ | rate GB/s or TFLOP/s | η | R² |
    |---|---|---|---|---|---|---|
    {chr(10).join(_fit_rows)}

    ("—" marks an overhead-dominated lane: its rate term is unidentifiable —
    the whole cost is t₀.) **Whether the parameters transfer is the open
    question.** Predicting each lane using only the *other* lanes of its
    class (leave-one-out; lanes with poor fits or unidentifiable rates may
    borrow but never lend):

    | machine | provider | phase | lanes pooled | median error | worst |
    |---|---|---|---|---|---|
    {chr(10).join(_loo_rows)}

    Where the probed ceiling is trustworthy the transfer is already usable
    (the two iGPU/discrete generation lanes predict each other within
    ~5–25%); the large errors trace to known probe gaps — the CPU bandwidth
    probe under-measures (one lane's measured decode *exceeds* its "ceiling"),
    and the compute probe measures f16 while inference runs quantized
    kernels. Fixing the probes and re-probing is the next step; after that,
    an Apple-silicon Mac is the held-out test — predicted from its ceilings
    first, then measured.
    """).text
    return (fleet_tab,)


@app.cell
def _(df, mo, probes):
    # ── Explore: machines & ceilings ────────────────────────────────────────
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

    explore_machines = mo.md(f"""
    ## The machines

    | machine | CPU | GPU | memory |
    |---|---|---|---|
    {chr(10).join(_spec_rows)}

    Before any model is loaded, the harness measures what each provider's
    silicon can do **bare**: how fast it multiplies matrices (the work of
    prompt reading) and how fast it moves memory (the work of token
    generation). Every model number in this report can be read against these
    ceilings, so a slow curve is attributable — runtime or device — rather
    than guessed at.

    | machine | provider | matrix math TFLOP/s | own memory GB/s | transfer-in GB/s |
    |---|---|---|---|---|
    {chr(10).join(_ceil_rows)}

    *Own-memory speed* is the device reading and writing its own memory (the
    STREAM convention, read+write counted). *Transfer-in* is data crossing
    into the device; on CPUs and unified memory the two are the same memory,
    so a gap between the columns is the PCIe link of a discrete card.
    """).text
    return (explore_machines,)


@app.cell
def _(charts, memory, mo, ok, pd, sweeps, switcher):
    # ── Explore: cost curves ────────────────────────────────────────────────
    # The job's ≈prompt length (backend-tokenized; the task doc is authored to
    # ~1700). Its decode runs at fills 1700→1956; the overlay sits mid-way.
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

    def _curve_panel(model):
        rows = _s[_s.model == model]
        over = _pre_overlay[_pre_overlay.model == model]
        html = mo.as_html(charts.curves(
            rows[rows.kind == "prefill"],
            "prompt reading: total time vs prompt length — ◆ = the validation "
            "job's time to first token",
            x="tokens", y="ttft_ms",
            x_title="prompt tokens", y_title="ms",
            overlay=over if len(over) else None)).text
        dec = rows[rows.kind == "decode"]
        dover = _dec_overlay[_dec_overlay.model == model]
        if len(dec):
            html += mo.as_html(charts.curves(
                dec.assign(kv_fill=lambda d: d.kv_fill.clip(lower=64)),
                "generation: tokens/s vs how full the context already is — "
                "◆ = the validation job",
                x="kv_fill", y="tps_p50", lo="tps_min", hi="tps_max",
                x_title="context already used (tokens)", y_title="tok/s",
                log_y=False, overlay=dover if len(dover) else None)).text
        memc = _mem[_mem.model == model] if len(_mem) else _mem
        if len(memc):
            html += mo.as_html(charts.curves(
                memc, "memory reserved for context, by context size",
                x="n_ctx", y="kv_mb",
                x_title="context (tokens)", y_title="MB")).text
        return html

    explore_curves = mo.md("""
    ## The cost curves

    Three questions, one chart each, per model. **Prompt reading**: how long
    until the model has read a prompt of a given length? (Log-log; a straight
    line means cost grows in step with length — an upward bend is attention
    getting quadratic.) **Generation**: how many tokens per second, and how
    much does a fuller context slow it? **Memory**: how much RAM does context
    itself cost? Lines are machine · provider; short lines ran out of sweep
    budget on slow silicon, not out of data.

    The **◆ diamonds** are not part of the sweep — they are the validation
    job, a real chat-templated workload, placed at its prompt length. A
    diamond on its line is the sweep predicting reality; a diamond off it is
    the synthetic-vs-real gap, shown rather than assumed away.
    """).text
    _models = sorted(_s.model.dropna().unique())
    explore_curves += (switcher.tabs(
        {m: _curve_panel(m) for m in _models}, group="curves")
        if _models else "")
    return JOB_TOKENS, explore_curves


@app.cell
def _(JOB_TOKENS, charts, df, memory, mo, ok, prep, sweeps, switcher, task_order):
    import math as _math

    # ── Explore: the validation job ─────────────────────────────────────────
    _counts = prep.status_cells(df)
    _bad_cells = _counts[_counts.status != "ok"]
    _ok_n, _all_n = int(_counts[_counts.status == "ok"].n.sum()), int(_counts.n.sum())
    _n_machines = df.machine.nunique()
    _verdict = (f"**{_ok_n}/{_all_n} jobs scored** across {_n_machines} machine"
                + ("s" if _n_machines != 1 else "") + ".")
    if len(_bad_cells):
        _verdict += " The misses: " + "; ".join(
            f"{r.n}× {r.who} {r.status.replace('_', ' ')}"
            for r in _bad_cells.itertuples()) + "."
    _bad_sweeps = df[df.sweep_status.isin(["too_slow", "errored"])]
    if len(_bad_sweeps):
        _verdict += " Sweeps that did not complete: " + "; ".join(
            f"{r.machine} · {r.provider} · {r.model} ({r.sweep_status.replace('_', ' ')})"
            for r in _bad_sweeps.itertuples()) + "."

    def _interp(points, at):
        pts = sorted(points)
        for (x0, y0), (x1, y1) in zip(pts, pts[1:], strict=False):
            if x0 <= at <= x1:
                f = (_math.log(at) - _math.log(x0)) / (_math.log(x1) - _math.log(x0))
                return _math.exp(_math.log(y0) + f * (_math.log(y1) - _math.log(y0)))
        return None

    _deltas = []
    for _r in ok.itertuples():
        _cs = sweeps[(sweeps.machine == _r.submission)
                     & (sweeps.provider == _r.provider)
                     & (sweeps.model == _r.model) & (sweeps.kind == "prefill")]
        _pred = _interp(list(zip(_cs.tokens, _cs.ttft_ms, strict=True)), JOB_TOKENS)
        if _pred and _r.ttft_ms_p50 == _r.ttft_ms_p50:
            _deltas.append(abs(_r.ttft_ms_p50 / _pred - 1) * 100)
    _agree = (f" Interpolating each lane's curve at the job's prompt length "
              f"puts the measured times a median "
              f"**{sorted(_deltas)[len(_deltas) // 2]:.0f}%** from the curve "
              f"(worst {max(_deltas):.0f}%)." if _deltas else "")

    explore_job = mo.md(f"""
    ## The validation job

    One real workload — `{task_order[0] if task_order else "—"}`, a
    ≈1.7k-token document summarized on a 256-token budget — run end to end on
    every lane. It exists to check that the curves predict reality, and it is
    the only place cold-start, load, and memory are measured. {_verdict}{_agree}

    Each bar below is one lane's full wall-clock from a cold process to a
    finished answer. Pale segments are setup paid once per process (loading
    weights, building the context, first-kernel warmup); vivid segments are
    the per-request work (reading the prompt, generating). Green total =
    fastest lane for that model.
    """).text

    _models = sorted(df.model.dropna().unique())
    _frames = {_m: prep.time_phases(ok[ok.model == _m]) for _m in _models}
    _nonempty = [f for f in _frames.values() if len(f)]
    _order = prep.shared_config_order(_nonempty) if _nonempty else None
    explore_job += switcher.tabs({
        _m: mo.as_html(charts.stacked(
            _frames[_m], charts.TIME_COLORS,
            "time to a finished answer (ms) — lower is better",
            dnf=prep.failures(df[df.model == _m]), config_order=_order)).text
        for _m in _models}, group="time") if _models else ""

    _mem = memory.assign(config=memory.machine + " · " + memory.provider) \
        if len(memory) else memory
    _jobm = ok.assign(config=ok.submission + " · " + ok.provider)
    _bars, _ticks = prep.memory_model(_mem, _jobm, at=2048)
    explore_job += mo.md("""
    Memory, said two ways: the **bars** are what the runtime's allocator
    reserves (weights, context, working space) at the job's context size; the
    **tick** is the whole process footprint an outside sampler measured while
    it generated. The gap between them is everything the allocator doesn't
    own — runtime, tokenizer, host copies. Shown, not modelled away.
    """).text
    if len(_bars):
        explore_job += mo.as_html(charts.stacked(
            _bars, charts.MEMORY_COLORS,
            "memory at the job's context (MB): bars = allocator, "
            "tick = measured footprint", ticks=_ticks)).text

    import html as _html
    def _quote(row):
        _who = f"{row.machine} · {row.model} {row.quant} · {row.provider}"
        return (f'<blockquote><span class="who">{_html.escape(_who)}</span>'
                f'{_html.escape(str(row.sample_completion).strip())}</blockquote>')
    _with_text = ok[ok.sample_completion.notna()].groupby("model", observed=True).head(1)
    explore_job += ('<details class="sample-completions">'
                    '<summary>generated summaries, one per model — proof the '
                    'timed runs were writing real text</summary>'
                    + "".join(_quote(r) for r in _with_text.itertuples())
                    + "</details>")
    return (explore_job,)


@app.cell
def _(mo, ok, prep):
    # ── Explore: GPU vs CPU ─────────────────────────────────────────────────
    gvc = prep.gpu_vs_cpu(ok)
    _asym = (f"the GPU reads prompts "
             f"**×{gvc.prefill_x.min():.1f}–{gvc.prefill_x.max():.0f}** faster "
             f"but generates only **×{gvc.decode_x.min():.1f}–"
             f"{gvc.decode_x.max():.1f}** faster") if len(gvc) else ""
    _rows = [f"| {_m} | {_s.provider_gpu.iloc[0]} | ×{_s.prefill_x.mean():.1f} "
             f"| ×{_s.decode_x.mean():.1f} | ×{_s.completion_x.mean():.1f} |"
             for _m, _s in gvc.groupby("machine", observed=True)]

    explore_gvc = mo.md(f"""
    ## GPU vs CPU on the same silicon

    What does falling back to the CPU cost? The two phases stress different
    hardware: reading the prompt is parallel math (the GPU's home game);
    generating tokens is bounded by memory speed, where CPUs are closer than
    their reputation. Measured across these machines: {_asym}. A fallback
    doesn't slow everything equally — the prompt side moves most.

    | machine | GPU | prompt reading | generation | full turn |
    |---|---|---|---|---|
    {chr(10).join(_rows)}
    """).text
    return (explore_gvc,)


@app.cell
def _(explore_curves, explore_gvc, explore_job, explore_machines, fleet_tab,
      mo, models_tab, switcher):
    mo.Html(switcher.tabs({
        "Models": models_tab,
        "Fleet": fleet_tab,
        "Explore": explore_machines + explore_curves + explore_job + explore_gvc,
    }, group="main"))
    return


if __name__ == "__main__":
    app.run()
