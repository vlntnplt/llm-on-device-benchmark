"""Build the report: one self-contained HTML file, no runtime dependencies.

The tested package computes everything (`load`/`prep`/`estimate`/`charts`);
this module turns it into template context and renders `templates/*.j2` with
the assets and vega libraries inlined. Charts are vega-lite specs embedded as
JSON islands; `assets/site.js` mounts them and drives the tabs; the fleet
calculator (`assets/fleet.js`) reads its coefficients from another island.

    uv run --project analysis python -m bench_analysis.site   # writes the default
    uv run --project analysis python -m bench_analysis.site --out path.html

Vega/vega-lite/vega-embed are fetched once at build time — versions pinned to
the installed altair's own constants — and cached under `third_party/vega/`
(untracked, like every build-time dependency in this repo). Nothing is fetched
when the page is *viewed*.
"""

from __future__ import annotations

import argparse
import html
import json
import math
import urllib.request
from pathlib import Path

import altair as alt
import pandas as pd
from jinja2 import Environment, PackageLoader, select_autoescape

from . import charts, estimate, load_memory, load_probes, load_results, load_sweeps, prep

PKG = Path(__file__).parent
REPO = PKG.parents[1]  # …/analysis/bench_analysis → the repo root
JOB_TOKENS = 1700  # the validation job's ≈prompt length (backend-tokenized)

VEGA_LIBS = (
    ("vega", alt.VEGA_VERSION),
    ("vega-lite", alt.VEGALITE_VERSION),
    ("vega-embed", alt.VEGAEMBED_VERSION),
)


def _vega_js(cache: Path) -> str:
    """The three vega libraries, concatenated; fetched once into `cache`."""
    parts = []
    cache.mkdir(parents=True, exist_ok=True)
    for name, version in VEGA_LIBS:
        f = cache / f"{name}@{version}.min.js"
        if not f.exists():
            url = f"https://cdn.jsdelivr.net/npm/{name}@{version}"
            with urllib.request.urlopen(url) as r:  # noqa: S310 — pinned https url
                f.write_bytes(r.read())
        parts.append(f.read_text())
    return "\n".join(parts)


def _spec(chart) -> str:
    return json.dumps(chart.to_dict())


def _interp_log(pts: list[tuple[float, float]], at: float) -> float | None:
    pts = sorted(pts)
    if not pts or not (pts[0][0] <= at <= pts[-1][0]):
        return None
    for (x0, y0), (x1, y1) in zip(pts, pts[1:], strict=False):
        if x0 <= at <= x1:
            f = math.log(at / x0) / math.log(x1 / x0)
            return y0 * (y1 / y0) ** f
    return None


def _models_ctx(df, sweeps, memory) -> list[dict]:
    costs = estimate.model_costs(df, memory)

    def fits(r, gb):
        need = r.file_bytes / 2**30 + r.kv_state_mb / 1024 + r.kv_slope_mb * 4096 / 1024 + 1.0
        return need <= gb

    out = []
    for r in costs.itertuples():
        anchors = []
        for (m, p), g in sweeps[sweeps.model == r.model].groupby(["machine", "provider"]):
            dec = g[(g.kind == "decode") & g.tps_p50.gt(0)]
            pre = g[(g.kind == "prefill") & g.ttft_ms.notna()]
            if not len(dec) and not len(pre):
                continue
            curve = list(zip(pre.tokens, pre.ttft_ms, strict=True))
            t2k = _interp_log(curve, 2048) if len(pre) else None
            t8k = _interp_log(curve, 8192) if len(pre) else None
            deep = dec.loc[dec.kv_fill.idxmax()] if len(dec) else None
            anchors.append({
                "lane": f"{m} · {p}",
                "t2k": f"{t2k / 1e3:.1f} s" if t2k else "—",
                "t8k": f"{t8k / 1e3:.1f} s" if t8k else "—",
                "fresh": f"{dec.loc[dec.kv_fill.idxmin()].tps_p50:.0f}" if len(dec) else "—",
                "deep": f"{deep.tps_p50:.0f} @ {deep.kv_fill / 1024:.0f}k"
                        if deep is not None else "—",
            })
        bad = df[(df.model == r.model) & (df.status != "ok")]
        caveats = "; ".join(
            f"{b.lane}: job {b.status.replace('_', ' ')}" for b in bad.itertuples())
        out.append({
            "model": r.model, "quant": r.quant,
            "file_gb": r.file_bytes / 2**30, "body_gb": r.body_bytes / 2**30,
            "kv_mb_per_1k": r.kv_slope_mb * 1024, "kv_state_mb": r.kv_state_mb,
            "fits8": fits(r, 8), "fits16": fits(r, 16),
            "anchors": anchors, "caveats": caveats,
        })
    return out


def _explore_ctx(df, ok, sweeps, memory, probes, task_order, specs: dict) -> dict:
    ctx: dict = {}
    ctx["specs"] = [
        {"machine": r.machine, "cpu": r.cpu, "gpu": r.gpu,
         "ram": (f"{r.ram_gb:g} GB" if r.ram_gb == r.ram_gb else "?")
                + (f", {int(r.ram_channels)}-ch @ {int(r.ram_mts)} MT/s"
                   if r.ram_channels == r.ram_channels and r.ram_channels else "")}
        for r in df.drop_duplicates("submission").itertuples()
    ]
    ctx["ceilings"] = []
    for (m, p), g in probes[probes.status == "ok"].groupby(["machine", "provider"]):
        copies = {r.kind: r.gbs for r in g[g.kind != "gemm"].itertuples()}
        ctx["ceilings"].append({
            "machine": m, "provider": p,
            "gemm": f"{g[g.kind == 'gemm'].tflops.max():.1f}",
            "d2d": f"{copies.get('d2d', float('nan')):.0f}",
            "h2d": f"{copies.get('h2d', float('nan')):.1f}",
        })

    # Cost curves, one panel of specs per model.
    s = sweeps.copy()
    s["config"] = s.machine + " · " + s.provider
    mem = memory.assign(config=memory.machine + " · " + memory.provider) \
        if len(memory) else memory
    job = ok.assign(config=ok.submission + " · " + ok.provider)
    pre_overlay = pd.DataFrame({"config": job.config, "model": job.model,
                                "x": JOB_TOKENS, "y": job.ttft_ms_p50}).dropna()
    dec_overlay = pd.DataFrame({"config": job.config, "model": job.model,
                                "x": JOB_TOKENS + 128, "y": job.decode_tps_p50}).dropna()
    ctx["curve_models"] = sorted(s.model.dropna().unique())
    ctx["curves"] = {}
    for model in ctx["curve_models"]:
        ids = []
        rows = s[s.model == model]
        over = pre_overlay[pre_overlay.model == model]
        sid = f"spec-curve-pre-{len(specs)}"
        specs[sid] = _spec(charts.curves(
            rows[rows.kind == "prefill"],
            "prompt reading: total time vs prompt length — ◆ = the validation job",
            x="tokens", y="ttft_ms", x_title="prompt tokens", y_title="ms",
            overlay=over if len(over) else None))
        ids.append(sid)
        dec = rows[rows.kind == "decode"]
        if len(dec):
            dover = dec_overlay[dec_overlay.model == model]
            sid = f"spec-curve-dec-{len(specs)}"
            specs[sid] = _spec(charts.curves(
                dec.assign(kv_fill=lambda d: d.kv_fill.clip(lower=64)),
                "generation: tokens/s vs context already used — ◆ = the validation job",
                x="kv_fill", y="tps_p50", lo="tps_min", hi="tps_max",
                x_title="context already used (tokens)", y_title="tok/s",
                log_y=False, overlay=dover if len(dover) else None))
            ids.append(sid)
        memc = mem[mem.model == model] if len(mem) else mem
        if len(memc):
            sid = f"spec-curve-mem-{len(specs)}"
            specs[sid] = _spec(charts.curves(
                memc, "memory reserved for context, by context size",
                x="n_ctx", y="kv_mb", x_title="context (tokens)", y_title="MB"))
            ids.append(sid)
        ctx["curves"][model] = ids

    # The validation job.
    counts = prep.status_cells(df)
    bad = counts[counts.status != "ok"]
    ok_n, all_n = int(counts[counts.status == "ok"].n.sum()), int(counts.n.sum())
    n_mach = df.machine.nunique()
    verdict = (f"<strong>{ok_n}/{all_n} jobs scored</strong> across {n_mach} machine"
               + ("s" if n_mach != 1 else "") + ".")
    if len(bad):
        verdict += " The misses: " + "; ".join(
            f"{r.n}× {html.escape(str(r.who))} {r.status.replace('_', ' ')}"
            for r in bad.itertuples()) + "."
    bad_sweeps = df[df.sweep_status.isin(["too_slow", "errored"])]
    if len(bad_sweeps):
        verdict += " Sweeps that did not complete: " + "; ".join(
            f"{r.machine} · {r.provider} · {r.model} ({r.sweep_status.replace('_', ' ')})"
            for r in bad_sweeps.itertuples()) + "."
    ctx["verdict"] = verdict

    deltas = []
    for r in ok.itertuples():
        cs = sweeps[(sweeps.machine == r.submission) & (sweeps.provider == r.provider)
                    & (sweeps.model == r.model) & (sweeps.kind == "prefill")]
        pred = _interp_log(list(zip(cs.tokens, cs.ttft_ms, strict=True)), JOB_TOKENS)
        if pred and r.ttft_ms_p50 == r.ttft_ms_p50:
            deltas.append(abs(r.ttft_ms_p50 / pred - 1) * 100)
    ctx["agreement"] = (
        f" Interpolating each lane's curve at the job's prompt length puts the "
        f"measured times a median <strong>{sorted(deltas)[len(deltas) // 2]:.0f}%</strong> "
        f"from the curve (worst {max(deltas):.0f}%)." if deltas else "")
    ctx["job_name"] = task_order[0] if task_order else "—"

    models = sorted(df.model.dropna().unique())
    frames = {m: prep.time_phases(ok[ok.model == m]) for m in models}
    nonempty = [f for f in frames.values() if len(f)]
    order = prep.shared_config_order(nonempty) if nonempty else None
    ctx["time_models"], ctx["time_specs"] = [], {}
    for m in models:
        if not len(frames[m]):
            continue
        sid = f"spec-time-{len(specs)}"
        specs[sid] = _spec(charts.stacked(
            frames[m], charts.TIME_COLORS,
            "time to a finished answer (ms) — lower is better",
            dnf=prep.failures(df[df.model == m]), config_order=order))
        ctx["time_models"].append(m)
        ctx["time_specs"][m] = sid

    bars, ticks = prep.memory_model(mem, job, at=2048)
    ctx["memory_spec"] = None
    if len(bars):
        sid = f"spec-mem-{len(specs)}"
        specs[sid] = _spec(charts.stacked(
            bars, charts.MEMORY_COLORS,
            "memory at the job's context (MB): bars = allocator, tick = measured footprint",
            ticks=ticks))
        ctx["memory_spec"] = sid

    # Completions are model output — the one untrusted string on the page.
    with_text = ok[ok.sample_completion.notna()].groupby("model", observed=True).head(1)
    ctx["completions"] = [
        {"who": html.escape(f"{r.machine} · {r.model} {r.quant} · {r.provider}"),
         "text": html.escape(str(r.sample_completion).strip())}
        for r in with_text.itertuples()
    ]

    gvc = prep.gpu_vs_cpu(ok)
    ctx["gvc_asym"] = (
        f"the GPU reads prompts <strong>×{gvc.prefill_x.min():.1f}–"
        f"{gvc.prefill_x.max():.0f}</strong> faster but generates only "
        f"<strong>×{gvc.decode_x.min():.1f}–{gvc.decode_x.max():.1f}</strong> faster"
        if len(gvc) else "")
    ctx["gvc"] = [
        {"machine": m, "gpu": g.provider_gpu.iloc[0],
         "prefill": f"×{g.prefill_x.mean():.1f}", "decode": f"×{g.decode_x.mean():.1f}",
         "completion": f"×{g.completion_x.mean():.1f}"}
        for m, g in gvc.groupby("machine", observed=True)
    ]
    return ctx


def _fleet_ctx(df, sweeps, memory, probes) -> dict:
    pts = estimate.points(df, sweeps, memory, probes)
    fits = estimate.lane_fits(pts)
    return {
        "fits": [
            {"machine": r.machine, "provider": r.provider, "kind": r.kind,
             "t0_ms": r.t0_ms,
             "rate": f"{r.rate:.0f}" if math.isfinite(r.rate) else "—",
             "eta": f"{r.eta:.2f}" if math.isfinite(r.eta) else "—",
             "r2": r.r2}
            for r in fits.itertuples()
        ],
        "loo": list(estimate.loo(pts).itertuples()),
        "fleet_data": json.dumps(
            estimate.fleet_coefficients(df, sweeps, memory, probes), indent=1),
    }


def build(published: Path, out: Path, vega_cache: Path | None = None) -> None:
    df, task_order = prep.prepare(load_results(published))
    sweeps = load_sweeps(published)
    probes = load_probes(published)
    memory = load_memory(published)
    ok = df[df.status == "ok"].copy()

    specs: dict[str, str] = {}
    env = Environment(loader=PackageLoader("bench_analysis"),
                      autoescape=select_autoescape(default=False))
    context = {
        "models": _models_ctx(df, sweeps, memory),
        **_explore_ctx(df, ok, sweeps, memory, probes, task_order, specs),
        **_fleet_ctx(df, sweeps, memory, probes),
        "specs": specs,
        "css": (PKG / "assets" / "site.css").read_text(),
        "site_js": (PKG / "assets" / "site.js").read_text(),
        "fleet_js": (PKG / "assets" / "fleet.js").read_text(),
        "vega_js": _vega_js(vega_cache or REPO / "third_party" / "vega"),
    }
    out.write_text(env.get_template("report.html.j2").render(context))


def main() -> None:
    ap = argparse.ArgumentParser(description="build the static report")
    ap.add_argument("--published", type=Path, default=REPO / "results" / "published")
    ap.add_argument("--out", type=Path, default=REPO / "results" / "published" / "report.html")
    args = ap.parse_args()
    build(args.published, args.out)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
