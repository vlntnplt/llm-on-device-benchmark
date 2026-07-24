"""Cross-machine benchmark — exploration & comparison.

A marimo notebook (a plain .py file — no jupytext pairing, diffs cleanly). The
tested package does the work: `bench_analysis.load` builds the tidy frame,
`bench_analysis.prep` derives the views, `bench_analysis.charts` owns the visual
language. This notebook just arranges them and writes the prose — it is the
disposable part.

    uv run --project analysis marimo edit analysis/report.py            # interactive
    uv run --project analysis marimo export html analysis/report.py -o report.html

Export to HTML to hand the team a static snapshot; add `--no-include-code` so it
reads as a report rather than a notebook. The export has no kernel, so nothing
here may depend on a reactive widget — a `mo.ui` element cannot switch anything
in it, and clicking one raises "Static notebook: this notebook is not connected
to a kernel". Every switch in the report is therefore a `bench_analysis.switcher`
group: panels pre-rendered for each context size and each backend filter, with
plain radios and CSS doing the switching.

Five sections, each a short claim backed by a chart, every number computed live
from the published submissions:

    1. time to an answer, stacked by phase (where does the time go?)
    2. memory while decoding (what must the device fit?)
    3. backend reliability across machines (which backend always answers?)
    4. ggml vs tjs lane by lane (same silicon — what does the backend cost?)
    5. ggml GPU vs CPU on the same silicon (what does a CPU fallback cost?)
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

    from bench_analysis import charts, load_results, prep, switcher

    return Path, charts, load_results, mo, pd, prep, switcher


@app.cell
def _(mo):
    mo.md("""
    # Cross-machine on-device LLM benchmark

    Two inference stacks run the same models, the same prompts, and the same
    token budgets, on each contributor's machine:

    - **ggml** — [llama.cpp](https://github.com/ggml-org/llama.cpp), a native
      C++ stack consuming GGUF files;
    - **tjs** — [Transformers.js](https://github.com/huggingface/transformers.js)
      running ONNX models on onnxruntime-node.

    Each measurement launches a fresh process, loads the model, feeds it a
    summarization prompt, and greedily decodes a fixed number of tokens. The
    prompt comes in three sizes — roughly 400, 850, and 1700 tokens — so each
    chart can show how cost grows with context. A **config** is one way of
    running a model: a machine, a backend, a compute provider (`cpu`, `cuda`,
    `metal`, …), and a quantization.

    Two reading notes. A quantization *label* is not the same math in both
    stacks (`ggml q4` is Q4_K_M, `tjs q4` is MatMulNBits), so comparisons hold
    *within* a label, never across labels. And every number on this page —
    including the ones in the text — is computed from the published submissions
    in `results/published/`, so the prose updates as new runs land.
    """)
    return


@app.cell
def _(Path, load_results, mo, pd, prep):
    # Anchor to the notebook's own location (…/analysis) so the path holds no matter
    # the cwd `marimo edit` was launched from; results live one dir up.
    _nb = mo.notebook_dir()
    _root = (_nb.parent / "results") if _nb else Path("../results")

    # The published submissions are the shared baseline: every PR-merged run under
    # results/published/<name>/.
    _df = load_results(_root / "published")

    # Preview a not-yet-published run alongside the baseline: point PREVIEW at a
    # local results dir (what `bench run --out` wrote — e.g. _root / "my-box") to
    # fold it in before you `bench publish` it. It loads under its in-file host name
    # with a " · preview" tag so it's distinct from the published rows. None = off.
    PREVIEW = None
    if PREVIEW is not None:
        _prev = load_results(PREVIEW)
        _prev["machine"] = _prev["machine"].astype(str) + " · preview"
        _df = pd.concat([_df, _prev], ignore_index=True)

    df, task_order = prep.prepare(_df)
    ok = df[df.status == "ok"].copy()

    _machines = df[["machine", "submission"]].drop_duplicates().sort_values("machine")
    mo.md("**Machines:** " + " · ".join(
        f"{r.machine} (`{r.submission}`)" for r in _machines.itertuples()))
    return df, ok, task_order


@app.cell
def _(mo, switcher):
    # tjs is slow and heavy enough to set the scale in §1 and §2, compressing the
    # ggml configs against the axis. This drops it from those charts so the
    # remaining spread is readable. §4 and §5 are unaffected — they exist to
    # compare the two backends, so there is nothing there to filter.
    mo.Html(switcher.backend_filter())
    return


@app.cell
def _(mo):
    mo.md("""
    ## 1 · Time to an answer

    How long from launching a cold process to a finished answer? Each bar is
    one config running one model; its length is that full wall-clock time,
    split into the five phases every run goes through, in order:

    - **model load** — read the weights from disk and place them on the device;
    - **context init** — allocate the KV cache and compute graph;
    - **warmup** — one minimal token pass that pays the one-time kernel/JIT
      compilation;
    - **prefill** — ingest the prompt; ends at the first generated token, so
      this segment is the time-to-first-token (TTFT);
    - **decode** — generate the answer, token by token.

    The pale segments are setup, paid once per process; the vivid ones are the
    generation work, paid on every request. The fastest config per model gets
    a green total. Configs that produced no usable number stay visible as
    full-width markers — amber when too slow to finish within the time budget,
    purple when a run crashed or ran out of memory.

    The tabs switch the prompt size. Rows keep one order across all tabs (the
    overall rank), so a config stays on its row while you flip between sizes —
    which can read slightly off a single tab's exact ranking. A stacked axis
    can't be log-scaled, so slow configs visually compress fast ones; the
    labels carry the exact totals, and §4 is the log-scale view where the
    small gaps become readable.
    """)
    return


@app.cell
def _(charts, df, mo, ok, prep, switcher, task_order):
    def _time_tabs(_ok, _df, group):
        # One row order pinned across the tabs, so a config never jumps rows when
        # the context size changes.
        _phases = {_t: prep.time_phases(_ok[_ok.task == _t]) for _t in task_order}
        _order = prep.shared_config_order(list(_phases.values()))
        return switcher.tabs({
            _t: mo.as_html(charts.stacked(
                _phases[_t], charts.TIME_COLORS,
                f"{_t}: time to a finished answer (ms) — "
                "load + prefill + decode, lower is better",
                dnf=prep.failures(_df[_df.task == _t]), config_order=_order)).text
            for _t in task_order
        }, group=group, active=task_order[len(task_order) // 2])

    # Both filter states are rendered here, not in the browser: the row order,
    # the axis scale, and the per-model winner are all computed over the rows a
    # chart shows, so dropping tjs client-side would leave all three stale.
    mo.Html(switcher.variants(
        _time_tabs(ok, df, "time-all"),
        _time_tabs(ok[ok.backend == "ggml"], df[df.backend == "ggml"], "time-ggml"),
    ))
    return


@app.cell
def _(mo):
    mo.md("""
    ## 2 · Memory while decoding

    What does generation occupy while it runs, and what must the device be
    able to hold? The solid segments are the **sustained** working set — the
    median RAM (and VRAM, where the provider uses a separate GPU pool) sampled
    while tokens were being generated: weights, KV cache, activations. The
    pale segment tops the bar up to the highest value seen during decode, so
    the bar total is the **peak** the device had to fit at some point.

    The two differ when a transient is still draining as decode starts — for
    example a provider that compiles kernels during the first full-context
    prefill. A missing VRAM segment means the provider has no separate VRAM
    pool to report (CPU runs, unified memory), not missing data. Prefill
    itself can peak higher than decode; §4's footprint view uses the
    high-water mark across both phases.
    """)
    return


@app.cell
def _(charts, mo, ok, prep, switcher, task_order):
    def _memory_tabs(_ok, group):
        _phases = {_t: prep.memory_phases(_ok[_ok.task == _t]) for _t in task_order}
        _order = prep.shared_config_order(list(_phases.values()))
        return switcher.tabs({
            _t: mo.as_html(charts.stacked(
                _phases[_t], charts.MEMORY_COLORS,
                f"{_t}: decode footprint (MB): solid = sustained RAM+VRAM, "
                "pale = transient peak — lower is better",
                config_order=_order)).text
            for _t in task_order
        }, group=group, active=task_order[len(task_order) // 2])

    mo.Html(switcher.variants(
        _memory_tabs(ok, "memory-all"),
        _memory_tabs(ok[ok.backend == "ggml"], "memory-ggml"),
    ))
    return


@app.cell
def _(df, prep):
    # Reliability, computed not asserted. Unhealthy configs (failed brain-check)
    # never ran a task, so they're listed apart from the attempted-cell tally.
    counts = prep.status_cells(df)
    n_machines = df.machine.nunique()

    _lines = []
    for _b, _sub in counts.groupby("backend"):
        _tasked = _sub[_sub.status != "unhealthy"]
        _ok_n, _n = int(_tasked[_tasked.status == "ok"].n.sum()), int(_tasked.n.sum())
        _miss = _tasked[_tasked.status != "ok"]
        _miss_txt = ("; misses: " + ", ".join(
            f"{r.n}× {r.provider} {r.status.replace('_', ' ')}"
            for r in _miss.itertuples())) if len(_miss) else ""
        _lines.append(f"- **{_b}: {_ok_n}/{_n} cells ok**{_miss_txt}.")
    _unh = counts[counts.status == "unhealthy"]
    if len(_unh):
        _lines.append("- failed the brain-check (tasks never ran): " + ", ".join(
            f"{r.n}× {r.who}" for r in _unh.itertuples()) + ".")
    reliability = "\n".join(_lines)
    return counts, n_machines, reliability


@app.cell
def _(mo, n_machines, reliability):
    mo.md(f"""
    ## 3 · Backend reliability across machines

    Speed only matters if a config produces an answer at all. Same models,
    same prompts, {n_machines} machines: a **cell** is one attempt — one
    (machine, model, prompt size) handed to one backend·provider. The tally
    counts how every attempted cell ended: `ok` (produced timed samples),
    `too slow` (hit the time budget, or ran below a usable tokens-per-second
    floor), `errored` (crashed or ran out of memory). Configs that failed the
    three-question sanity gate are listed apart — their timed cells never ran.

    {reliability}
    """)
    return


@app.cell
def _(charts, counts, mo, switcher):
    mo.Html(switcher.variants(
        mo.as_html(charts.status_bars(counts)).text,
        mo.as_html(charts.status_bars(counts[counts.backend == "ggml"])).text,
    ))
    return


@app.cell
def _(df, ok, prep):
    lane_t = prep.lane_time(ok)
    lane_m = prep.lane_memory(ok)
    # A lane holds either the cpu provider or accelerated ones, never both.
    lane_t["kind"] = lane_t.provider.eq("cpu").map({True: "cpu", False: "gpu"})
    lane_m["kind"] = lane_m.provider.eq("cpu").map({True: "cpu", False: "gpu"})

    # tjs/ggml ratios per (lane, model, task) cell where both backends answered.
    def _h2h(frame, col):
        w = frame.pivot_table(index=["lane", "model", "task"], columns="backend",
                              values=col, observed=True)
        if not {"ggml", "tjs"} <= set(w.columns):
            return None
        r = (w.tjs / w.ggml).dropna()
        return r if len(r) else None

    # The headline gaps, only claimed as universal when the data says so.
    _rt = _h2h(lane_t, "total_s")
    h2h_gap = ""
    if _rt is not None:
        _every = (" — in every lane, model, and context size"
                  if _rt.min() > 1 else "")
        h2h_gap = (f"ggml gets to the answer **×{_rt.min():.1f}–×{_rt.max():.1f} "
                   f"faster** (median ×{_rt.median():.1f}){_every}.")

    _rm = _h2h(lane_m, "peak_gb")
    h2h_mem = ""
    if _rm is not None:
        h2h_mem = (f"tjs's lightest config peaks at "
                   f"**×{_rm.min():.1f}–×{_rm.max():.1f}** ggml's footprint "
                   f"(median ×{_rm.median():.1f}).")

    # The summary split the pooled headline hides: the backend gap is a very
    # different story on a CPU than on a GPU. Mean over matchup cells, like §5.
    def _x(r):
        return f"×{r.mean():.1f}" if r is not None else "—"

    h2h_table = "\n".join(
        f"| {_kind} lanes | {_x(_h2h(lane_t[lane_t.kind == _kind], 'total_s'))} "
        f"| {_x(_h2h(lane_m[lane_m.kind == _kind], 'peak_gb'))} |"
        for _kind in ("cpu", "gpu"))

    # Which provider each backend's winner used in the GPU lanes: one provider
    # everywhere is a deployable story; a different one per machine is not.
    # (GPU-lane rows are exactly the non-cpu-provider rows, by construction.)
    _gpu = lane_t[lane_t.provider != "cpu"]

    def _providers(b):
        s = (_gpu[_gpu.backend == b].groupby("lane", observed=True)
             .provider.agg(lambda v: "/".join(sorted(set(v)))))
        return ", ".join(f"{m}: **{p}**" for m, p in s.items())

    h2h_providers = "\n".join(f"- fastest **{b}** provider — {_providers(b)}"
                              for b in sorted(_gpu.backend.unique()))

    # Walkover lanes (one backend never produced a usable sample there) are
    # dropped from the charts — a lone dot says nothing about the gap — and
    # disclosed here instead, with the failure mode pulled from the record.
    _solo_lanes = sorted(set(lane_t[lane_t.n_backends < 2].lane))
    _notes = []
    for _lane in _solo_lanes:
        _present = set(lane_t[lane_t.lane == _lane].backend)
        for _b in sorted(set(lane_t.backend.unique()) - _present):
            _st = df[(df.lane == _lane) & (df.backend == _b)].status
            _why = "/".join(sorted(set(_st[_st != "ok"].str.replace("_", " ")))
                            ) or "never ran"
            _notes.append(f"**{_lane}** — no usable {_b} sample ({_why})")
    h2h_note = ("Lanes without a matchup are left out of the charts: "
                + "; ".join(_notes) + ".") if _notes else ""
    return h2h_gap, h2h_mem, h2h_note, h2h_providers, h2h_table, lane_m, lane_t


@app.cell
def _(h2h_gap, h2h_note, h2h_providers, h2h_table, mo):
    mo.md(f"""
    ## 4 · ggml vs tjs, lane by lane

    A **lane** is one piece of silicon: a machine's CPU, or its GPU. Within a
    lane both backends ran on exactly the same hardware, so the distance
    between the two dots is attributable to the software stack — runtime,
    model format, kernels — not the machine. Each dot is that backend's
    fastest config for the model and prompt size; the small label on the
    right is the gap as a multiplier. The axis is logarithmic, so equal
    distances are equal *ratios* wherever they appear. The tabs switch the
    prompt size. {h2h_gap}

    Averaged over the cells where both backends answered (models × prompt
    sizes), split by the kind of silicon:

    | | time, tjs / ggml | peak footprint, tjs / ggml |
    |---|---|---|
    {h2h_table}

    A GPU-lane matchup is only as comparable as the providers that won it —
    each backend brings its own GPU path, so note which one each winner used:

    {h2h_providers}

    {h2h_note}
    """)
    return


@app.cell
def _(charts, lane_t, mo, switcher, task_order):
    _m = lane_t[lane_t.n_backends >= 2]
    mo.Html(switcher.tabs({
        _t: mo.as_html(charts.dumbbell(
            _m[_m.task == _t],
            f"{_t}: total time to a finished answer (s, log) — "
            "each backend's fastest config per lane")).text
        for _t in task_order
    }, group="h2h-time", active=task_order[len(task_order) // 2]))
    return


@app.cell
def _(h2h_mem, mo):
    mo.md(f"""
    The same lanes, by **peak footprint** — the highest RAM+VRAM the run
    touched across prefill *and* decode, transient spikes included: the
    memory a device must actually have free to run the config at all. Dots
    are each backend's lightest config. {h2h_mem}
    """)
    return


@app.cell
def _(charts, lane_m, mo, switcher, task_order):
    _m = lane_m[lane_m.n_backends >= 2]
    mo.Html(switcher.tabs({
        _t: mo.as_html(charts.dumbbell(
            _m[_m.task == _t],
            f"{_t}: peak footprint (GB, log) — high-water RAM+VRAM across "
            "prefill and decode, each backend's lightest config per lane",
            value="peak_gb", value_title="peak (GB)")).text
        for _t in task_order
    }, group="h2h-mem", active=task_order[len(task_order) // 2]))
    return


@app.cell
def _(ok, prep):
    gvc = prep.gpu_vs_cpu(ok)

    # One verdict row per machine: speedups averaged over models and context
    # sizes. The phase asymmetry headline is computed, not asserted — it holds
    # whatever runs land.
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
    ## 5 · ggml: GPU vs CPU on the same silicon

    On the machines that have both, what does running on the CPU instead of
    the GPU cost? The two generation phases stress different parts of the
    chip — prefill processes the whole prompt in parallel and is
    compute-bound; decode produces one token at a time and is bound by memory
    bandwidth — so the measured cost is lopsided: {gvc_asym}. A CPU fallback
    doesn't slow everything proportionally; the prompt-reading side moves
    most. One row per machine, speedups averaged over models and prompt
    sizes; **full turn** is prefill + decode together, the two phases
    weighted by where the time actually goes.

    | machine | GPU | prefill speedup | decode speedup | full-turn speedup |
    |---|---|---|---|---|
    {gvc_table}

    Below, the same comparison in absolute seconds, in §4's dumbbell
    language: per model, one dumbbell for time-to-first-token and one for
    decode time — **slate** cpu, **teal** gpu. The dashed rule marks one
    second, a common responsiveness reference: where a dot sits against it
    is what a user actually waits.
    """)
    return


@app.cell
def _(charts, mo, ok, prep, switcher, task_order):
    fc = prep.fallback_cost(ok)
    _ysort = [f"{_m} · {_p}" for _m in sorted(fc.model.unique())
              for _p in ("TTFT", "decode")]
    mo.Html(switcher.tabs({
        _t: mo.as_html(charts.dumbbell(
            fc[fc.task == _t],
            f"{_t}: the fallback tax (s, log) — TTFT and decode time per "
            "turn, cpu vs gpu; dashed = 1 s",
            row="machine", value="seconds", value_title="time (s)", width=420,
            y="leg", y_sort=_ysort, hue="side", colors=charts.CPU_GPU_COLORS,
            ref_x=1.0)).text
        for _t in task_order
    }, group="fallback", active=task_order[len(task_order) // 2]))
    return


if __name__ == "__main__":
    app.run()
