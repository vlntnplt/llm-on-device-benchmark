"""Build the report: one self-contained HTML file, no runtime dependencies.

The tested package computes everything (`load`/`prep`/`estimate`/`charts`);
this module turns it into template context and renders `templates/*.j2` with
the assets and vega libraries inlined. Charts are vega-lite specs embedded as
JSON islands; `assets/site.js` mounts them and drives the tabs; the fleet
calculator (`assets/fleet.js`) reads its coefficients from another island.

Lane identity is assigned here: every lane (machine × provider) that appears
anywhere on the page gets one slot from `charts.LANE_SLOTS`, and the same
domain→range scale goes to every curve chart while the Models tab wears the
matching CSS variable — one lane, one colour, everywhere.

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
from datetime import date
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


def _lane_namer(df):
    """(submission, provider) → a silicon-named lane label ("RTX 5080 ·
    vulkan", "Ryzen 9 9950X · cpu") — nobody should have to read slugs."""
    chips = prep._silicon(df.assign(machine=df.submission))

    def lane(sub: str, provider: str) -> str:
        cpu, gpu = chips.get(sub, (sub, ""))
        chip = cpu if provider == "cpu" else (gpu or cpu)
        return f"{chip} · {provider}"

    return lane


def _lane_slots(df, lane) -> dict[str, int]:
    """Every lane on the page → its fixed colour slot. Sorted for a stable
    assignment within a build; each rendered page is self-consistent."""
    lanes = sorted({lane(m, p) for m, p in
                    zip(df.submission, df.provider, strict=True)})
    return {label: i for i, label in enumerate(lanes)}


def _lane_scale(slots: dict[str, int]) -> alt.Scale:
    return alt.Scale(domain=list(slots),
                     range=[charts.lane_color(i) for i in slots.values()])


def _css_slot(i: int) -> str:
    """Slot index → the CSS custom-property class the templates use."""
    return f"s{i + 1}" if i < len(charts.LANE_SLOTS) else "sx"


def _fmt_s(ms: float | None) -> str:
    return f"{ms / 1e3:.1f} s" if ms else "—"


CLASS_ORDER = ("cpu", "igpu", "dgpu")
CLASS_LABELS = {"cpu": "CPU", "igpu": "integrated GPU", "dgpu": "discrete GPU"}


def _hw_classes(pairs, probes) -> dict[tuple[str, str], str]:
    """(submission, provider) → hardware class: "cpu", "igpu" or "dgpu".

    Classified from the bandwidth probes, not the device name: a discrete card
    has its own memory, measurably faster than its host's (RTX 5080: 816 vs
    35 GB/s d2d); an integrated GPU shares the host's and matches it. An
    accelerated lane with no probe reads as integrated — the claim that
    assumes the least extra hardware."""
    d2d = {(r.machine, r.provider): r.gbs
           for r in probes[(probes.kind == "d2d") & (probes.status == "ok")].itertuples()}
    out = {}
    for m, p in pairs:
        gpu, host = d2d.get((m, p)), d2d.get((m, "cpu"))
        out[(m, p)] = ("cpu" if p == "cpu" else
                       "dgpu" if gpu and host and gpu > 2 * host else "igpu")
    return out


def _rng(vals: list[float], fmt: str) -> str:
    """A min–max range, collapsed when the values agree at display precision."""
    if not vals:
        return "—"
    lo, hi = format(min(vals), fmt), format(max(vals), fmt)
    return lo if lo == hi else f"{lo}–{hi}"


def _rejects(fits, lane) -> dict[tuple[str, str], str]:
    """Lanes the ranges must not anchor on: decode fits the estimator refuses
    to pool (see `estimate.lenders`), each with the measured reason. Decode is
    the gate — a lane whose generation shows no bandwidth term never engaged
    the silicon, and none of its numbers describe the hardware class."""
    good = {(r.machine, r.provider)
            for r in estimate.lenders(fits).itertuples() if r.kind == "decode"}
    out = {}
    for r in fits[fits.kind == "decode"].itertuples():
        if (r.machine, r.provider) in good:
            continue
        if not math.isfinite(r.eta):
            flat = f" at ≈{1e3 / r.t0_ms:.0f} tok/s" if r.t0_ms > 0 else ""
            why = (f"generation is flat{flat} whatever the model — "
                   f"all overhead, no bandwidth term")
        else:
            why = f"the affine fit does not describe it (R² {r.r2:.2f})"
        out[(r.machine, r.provider)] = f"{lane(r.machine, r.provider)}: {why}"
    return out


def _models_ctx(df, sweeps, memory, lane, slots, classes, rejects) -> dict:
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
            fresh = float(dec.loc[dec.kv_fill.idxmin()].tps_p50) if len(dec) else None
            deep = dec.loc[dec.kv_fill.idxmax()] if len(dec) else None
            label = lane(m, p)
            cls = classes.get((m, p), "igpu")
            anchors.append({
                "lane": label, "slot": _css_slot(slots.get(label, len(slots))),
                "cls": CLASS_LABELS[cls], "cls_key": cls,
                "rejected": (m, p) in rejects,
                "t2k": _fmt_s(t2k), "t8k": _fmt_s(t8k), "t2k_v": t2k,
                "fresh": f"{fresh:.0f}" if fresh is not None else "—",
                "fresh_v": fresh,
                "deep": f"{deep.tps_p50:.0f} @ {deep.kv_fill / 1024:.0f}k"
                        if deep is not None else "—",
            })
        # Grouped by hardware class, alphabetical within — an ordering, not a
        # ranking.
        anchors.sort(key=lambda a: (CLASS_ORDER.index(a["cls_key"]), a["lane"]))
        ranges = {}
        for cls in CLASS_ORDER:
            of = [a for a in anchors if a["cls_key"] == cls and not a["rejected"]]
            fresh = [a["fresh_v"] for a in of if a["fresh_v"] is not None]
            t2k = [a["t2k_v"] / 1e3 for a in of if a["t2k_v"] is not None]
            ranges[cls] = {"fresh": _rng(fresh, ".0f"), "t2k": _rng(t2k, ".1f")}
        bad = df[(df.model == r.model) & (df.status != "ok")]
        caveats = "; ".join(
            f"{lane(b.submission, b.provider)}: job {b.status.replace('_', ' ')}"
            for b in bad.itertuples())
        out.append({
            "model": r.model, "quant": r.quant,
            "file_gb": r.file_bytes / 2**30, "body_gb": r.body_bytes / 2**30,
            "kv_mb_per_1k": r.kv_slope_mb * 1024, "kv_state_mb": r.kv_state_mb,
            "fits8": fits(r, 8), "fits16": fits(r, 16),
            "anchors": anchors, "ranges": ranges, "caveats": caveats,
        })

    # Class columns, described by the silicon whose measurements anchor them.
    members: dict[str, set] = {c: set() for c in CLASS_ORDER}
    for (m, p), cls in classes.items():
        if (m, p) not in rejects:
            members[cls].add(lane(m, p).rsplit(" · ", 1)[0])
    cols = [{"key": c, "label": CLASS_LABELS[c], "members": ", ".join(sorted(members[c]))}
            for c in CLASS_ORDER if members[c]]
    return {"models": out, "class_cols": cols,
            "rejects": sorted(rejects.values())}


def _evidence_ctx(df, ok, sweeps, memory, probes, task_order, specs: dict,
                  lane, lane_scale) -> dict:
    ctx: dict = {}
    display = dict(zip(df.submission, df.machine, strict=False))
    ctx["machine_rows"] = [
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
            "machine": display.get(m, m), "provider": p,
            "gemm": f"{g[g.kind == 'gemm'].tflops.max():.1f}",
            "d2d": f"{copies.get('d2d', float('nan')):.0f}",
            "h2d": f"{copies.get('h2d', float('nan')):.1f}",
        })

    # Cost curves, one panel of specs per model.
    s = sweeps.copy()
    s["config"] = [lane(m, p) for m, p in zip(s.machine, s.provider, strict=True)]
    mem = memory.assign(config=[lane(m, p) for m, p in
                                zip(memory.machine, memory.provider, strict=True)]) \
        if len(memory) else memory
    job = ok.assign(config=[lane(m, p) for m, p in
                            zip(ok.submission, ok.provider, strict=True)])
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
            hue_scale=lane_scale, overlay=over if len(over) else None))
        ids.append(sid)
        dec = rows[rows.kind == "decode"]
        if len(dec):
            dover = dec_overlay[dec_overlay.model == model]
            sid = f"spec-curve-dec-{len(specs)}"
            specs[sid] = _spec(charts.curves(
                dec.assign(kv_fill=lambda d: d.kv_fill.clip(lower=64)),
                "generation: tokens/s vs context already used (log–log; "
                "every lane keeps its shape) — ◆ = the validation job",
                x="kv_fill", y="tps_p50", lo="tps_min", hi="tps_max",
                x_title="context already used (tokens)", y_title="tok/s",
                hue_scale=lane_scale, overlay=dover if len(dover) else None))
            ids.append(sid)
        memc = mem[mem.model == model] if len(mem) else mem
        if len(memc):
            # The allocator ladder is machine-independent — every lane reports
            # the same points, so six overplotted lines would be a fake
            # six-series chart. Pool to one line, no legend.
            pooled = (memc.groupby(["model", "n_ctx"], as_index=False)
                      .kv_mb.median())
            sid = f"spec-curve-mem-{len(specs)}"
            specs[sid] = _spec(charts.curves(
                pooled, "memory reserved for context, by context size "
                        "(allocator ladder — identical on every machine)",
                x="n_ctx", y="kv_mb", x_title="context (tokens)", y_title="MB",
                hue=None))
            ids.append(sid)
        ctx["curves"][model] = ids

    # The validation job.
    counts = prep.status_cells(df)
    bad = counts[counts.status != "ok"]
    ok_n, all_n = int(counts[counts.status == "ok"].n.sum()), int(counts.n.sum())
    n_mach = df.machine.nunique()
    ctx["jobs_ok"], ctx["jobs_all"] = ok_n, all_n
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

    lane = _lane_namer(df)
    slots = _lane_slots(df, lane)
    pairs = (set(zip(df.submission, df.provider, strict=True))
             | set(zip(sweeps.machine, sweeps.provider, strict=True)))
    classes = _hw_classes(pairs, probes)
    fits = estimate.lane_fits(estimate.points(df, sweeps, memory, probes))
    rejects = _rejects(fits, lane)

    specs: dict[str, str] = {}
    env = Environment(loader=PackageLoader("bench_analysis"),
                      autoescape=select_autoescape(default=False))
    evidence = _evidence_ctx(df, ok, sweeps, memory, probes, task_order, specs,
                             lane, _lane_scale(slots))
    context = {
        **_models_ctx(df, sweeps, memory, lane, slots, classes, rejects),
        **evidence,
        **_fleet_ctx(df, sweeps, memory, probes),
        "stats": [
            {"v": f"{df.machine.nunique()}", "k": "machines"},
            {"v": f"{df.model.nunique()}", "k": "models"},
            {"v": f"{len(slots)}", "k": "lanes measured"},
            {"v": f"{evidence['jobs_ok']}/{evidence['jobs_all']}", "k": "jobs scored"},
        ],
        "built": date.today().strftime("%-d %B %Y"),
        "specs": specs,
        "dark_map": json.dumps(charts.DARK_MAP),
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
