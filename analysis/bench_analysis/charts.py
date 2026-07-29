"""Altair builders sharing one visual language.

The rules, in one place so every chart speaks them:

- **a lane keeps its colour everywhere** — `site.py` assigns each lane
  (machine × provider) a slot from the validated categorical palette below,
  and passes the same scale to every curve chart. Identity follows the lane,
  never the chart.
- **green is never a data series** — the success text tone marks an `ok`
  outcome, nothing else.
- failure modes keep one status hue everywhere (washes, table marks):
  amber `too_slow`, salmon `errored`, red `unhealthy` — always next to a
  text label, never colour alone.
- in phase stacks, recessive greys = setup, blue/teal = the request work
  (prefill/decode), so the eye lands on the part that answers the user; in
  the memory stack, slate = weights, blue = the KV cache that grows with
  context, pale = compute workspace.
- an INK tick laid over a computed stack is an independent *measured* value —
  the sampled footprint against the allocator's arithmetic.

Specs are rendered once with the light-mode colours; `assets/site.js` swaps
them for their dark-surface steps through `DARK_MAP` at mount time, so both
themes get colours picked for their surface rather than one compromise.
Both palettes and the tier ramp are validated (CVD separation, lightness
band, contrast) against the page's actual surfaces.
"""

from __future__ import annotations

import math

import altair as alt
import pandas as pd

from . import prep

# Categorical slots for lane identity, fixed order — assigned in sequence by
# site.py, never cycled. The dark list is the same hues re-stepped for the
# dark surface. Lanes past the validated eight all wear the overflow grey
# (the legend still names them); more lanes than that wants facets, not hues.
LANE_SLOTS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
              "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
LANE_SLOTS_DARK = ["#3987e5", "#d95926", "#199e70", "#c98500",
                   "#d55181", "#008300", "#9085e9", "#e66767"]
LANE_OVERFLOW = "#898781"


def lane_color(i: int) -> str:
    return LANE_SLOTS[i] if i < len(LANE_SLOTS) else LANE_OVERFLOW


ACCENT = "#006300"   # success *text*: ok
MUTED = "#8a939b"    # secondary labels, reference rules
INK = "#454e57"      # values the reader reads off a mark (ticks, totals)

# The page shows through transparent charts, so the "white doing the
# separating" — segment gaps, marker rings — is drawn in the page colour.
SURFACE = "#f6f8fa"

STATUS_COLORS = {"ok": "#0ca30c", "too_slow": "#fab219",
                 "errored": "#ec835a", "unhealthy": "#d03b3b"}

# Setup recedes (greys), the request work pops (blue prefill, teal decode).
TIME_COLORS = dict(zip(prep.TIME_PHASES,
                       ["#aab6c2", "#c3ccd6", "#dde3e9", "#2a78d6", "#1baf7a"],
                       strict=True))
# Slate = the weight bulk, blue = the KV cache that grows with context,
# pale = the compute workspace.
MEMORY_COLORS = dict(zip(prep.MEMORY_PHASES,
                         ["#64748b", "#2a78d6", "#c3ccd6"], strict=True))

# light hex → its dark-surface step; site.js walks every mounted spec with
# this when the page is dark. Greys flip to steps that recede on a dark
# surface instead of glowing on it.
DARK_MAP = {
    **dict(zip(LANE_SLOTS, LANE_SLOTS_DARK, strict=True)),
    ACCENT: "#0ca30c",
    INK: "#cdd3d9",
    SURFACE: "#11161b",
    "#aab6c2": "#5d6873", "#c3ccd6": "#4a545f", "#dde3e9": "#39424b",
    "#64748b": "#8b99a8",
}

_DNF_COLORS = {v: STATUS_COLORS[k] for k, v in prep.FAIL_LABELS.items()}


def _log_ticks(values: pd.Series) -> list[float] | None:
    """Ticks at 1-2-5 per decade over the data's range.

    A log axis defaults to a line at every integer, which on a two-decade span
    is a picket fence behind the marks. Anchoring to 1-2-5 keeps a reader's
    sense of scale with a third of the lines."""
    pool = [v for v in values.dropna() if v > 0]
    if not pool:
        return None
    lo, hi = min(pool), max(pool)
    ticks, decade = [], math.floor(math.log10(lo))
    while 10**decade <= hi:
        ticks += [m * 10**decade for m in (1, 2, 5)]
        decade += 1
    inside = [t for t in ticks if lo / 2 <= t <= hi * 2]
    return inside or None


def _themed(chart: alt.Chart) -> alt.Chart:
    """Let the page's background show through, so the chart sits on the page
    rather than punching a white panel out of it. Axis/legend/grid inks are
    applied at mount time from the page's own CSS tokens (site.js)."""
    return chart.configure(background="transparent")


def _captioned(chart: alt.Chart, text: str) -> alt.Chart:
    """One caption along the bottom of the whole chart — a per-facet axis title
    would repeat (and collide) under every facet column."""
    return chart.properties(
        title=alt.TitleParams(text, orient="bottom", anchor="middle",
                              fontWeight="normal", fontSize=12, dy=8))


def stacked(g: pd.DataFrame, colors: dict[str, str], caption: str, *,
            width: int = 560, fmt: str = ".0f",
            dnf: pd.DataFrame | None = None,
            config_order: list[str] | None = None,
            ticks: pd.DataFrame | None = None) -> alt.Chart:
    """Horizontal stacked bars, one row facet per model, ranked by total
    ascending. Row facets (not columns) so the chart never outgrows the page —
    a report scrolls down, not sideways.

    `g` is long: one row per (model, config, phase) with a numeric `value`.
    `colors` maps phase → fill; its dict order is the bottom→top draw order.
    Segments separate with a 2px surface-colour gap, never a drawn border.
    The total per config is labelled at the bar end.
    `dnf` (from `prep.failures`) adds the
    configs with no usable sample as full-width status washes sorted to the
    bottom; their text label carries the meaning, the wash only echoes it.
    `config_order` (e.g. from `prep.shared_config_order`) pins the row order —
    pass the same list to sibling charts so configs don't jump rows.
    `ticks` (model, config, value) lays one INK tick per config over the
    stack — an independently *measured* value against the computed bar.
    """
    order = list(colors)
    g = g[g.phase.isin(order)].copy()
    g["rank"] = g.phase.map({p: i for i, p in enumerate(order)})

    tot = g.groupby(["model", "config"], observed=True)["value"].sum().reset_index(name="total")
    g = g.merge(tot, on=["model", "config"])

    # Fastest/lightest first, shared across facets so rows line up; no-sample
    # configs sort to the bottom (worst). Configs the pinned order doesn't know
    # (e.g. failed in every task) append at the end.
    cfg_order = (list(config_order) if config_order is not None
                 else tot.groupby("config")["total"].min().sort_values().index.tolist())
    has_dnf = dnf is not None and len(dnf)
    has_ticks = ticks is not None and len(ticks)
    if has_dnf:
        cfg_order += [c for c in dnf.config.unique() if c not in cfg_order]
    if has_ticks:
        cfg_order += [c for c in ticks.config.unique() if c not in cfg_order]

    # A faceted layer chart must share ONE top-level dataset — Vega-Lite rejects
    # per-layer data under a facet. So stack the bar rows, the no-sample rows,
    # and the tick rows together and split them back per layer with filters on
    # `is_dnf` / `is_tick`.
    g["is_dnf"] = False
    g["is_tick"] = False
    frames = [g]
    if has_dnf:
        d = dnf.copy()
        d["x"] = float(tot["total"].max()) if len(tot) else 1.0
        d["mid"] = d["x"] / 2
        d["is_dnf"] = True
        d["is_tick"] = False
        frames.append(d)
    if has_ticks:
        t = ticks.copy()
        t["is_dnf"] = False
        t["is_tick"] = True
        frames.append(t)
    data = pd.concat(frames, ignore_index=True)
    # NaN in an object-dtype column (phase on dnf/tick rows, label on bar rows)
    # escapes Altair's sanitization and breaks the HTML export — blank it.
    for _col in ("phase", "label"):
        if _col in data:
            data[_col] = data[_col].fillna("")

    # Hardware-named configs run long ("Ryzen 9 9950X + RTX 5080 · ggml-vulkan");
    # the default ~180px label limit would clip them.
    y = alt.Y("config:N", title=None, sort=cfg_order,
              axis=alt.Axis(labelLimit=260))
    base = alt.Chart(data)
    bars = base.transform_filter("!datum.is_dnf && !datum.is_tick").mark_bar(
        stroke=SURFACE, strokeWidth=2).encode(
        y,
        x=alt.X("value:Q", title=None, stack=True),
        color=alt.Color("phase:N", title=None, sort=order,
                        scale=alt.Scale(domain=order, range=[colors[p] for p in order]),
                        legend=alt.Legend(orient="bottom")),
        order=alt.Order("rank:Q"),
        tooltip=["model", "config", "phase",
                 alt.Tooltip("value:Q", title="this segment", format=fmt),
                 alt.Tooltip("total:Q", format=fmt)],
    )
    # One total label per config: filter to the first phase (every config has it).
    labels = base.transform_filter(
        f"!datum.is_dnf && !datum.is_tick && datum.phase == '{order[0]}'"
    ).mark_text(align="left", dx=6, fontWeight="bold").encode(
        y, x=alt.X("total:Q"), text=alt.Text("total:Q", format=fmt),
        color=alt.value(INK),
    )
    layers = [bars, labels]

    if has_ticks:
        layers.append(base.transform_filter("datum.is_tick").mark_tick(
            color=INK, thickness=2.5, size=20).encode(
            y, x=alt.X("value:Q"),
            tooltip=["model", "config",
                     alt.Tooltip("value:Q", title="measured", format=fmt)]))

    if has_dnf:
        dom = list(_DNF_COLORS)
        cscale = alt.Scale(domain=dom, range=[_DNF_COLORS[k] for k in dom])
        dbase = base.transform_filter("datum.is_dnf")
        layers += [
            dbase.mark_bar(opacity=0.2).encode(
                y, x=alt.X("x:Q", title=None),
                color=alt.Color("label:N", scale=cscale, legend=None),
                tooltip=["model", "config", alt.Tooltip("label:N", title="status")],
            ),
            # The label is text-ink, not status-colour: amber/salmon text fails
            # contrast on the light page; the wash carries the hue instead.
            dbase.mark_text(fontWeight="bold", color=INK).encode(
                y, x=alt.X("mid:Q"), text=alt.Text("label:N"),
            ),
        ]

    # Step-sized rows: each facet is exactly as tall as the configs it holds.
    # Model names ride above each panel rather than in a rotated left gutter.
    return _themed(_captioned(
        alt.layer(*layers).properties(width=width, height=alt.Step(24)).facet(
            row=alt.Row("model:N", title=None,
                        header=alt.Header(labelOrient="top", labelAnchor="start",
                                          labelAngle=0, labelFontWeight="bold",
                                          labelFontSize=13))),
        caption))


def curves(pts: pd.DataFrame, caption: str, *,
           x: str, y: str, lo: str | None = None, hi: str | None = None,
           x_title: str, y_title: str, hue: str | None = "config",
           hue_scale: alt.Scale | None = None,
           log_x: bool = True, log_y: bool = True, width: int = 620,
           overlay: pd.DataFrame | None = None) -> alt.Chart:
    """Measured curves: one 2px line + ≥8px points per `hue` value, faceted by
    model. `hue_scale` pins each lane to its site-wide colour slot — pass the
    same scale to every curves chart so a lane never changes colour between
    panels. `hue=None` draws a single slot-1 series with no legend (one
    series needs no box — the caption names it). `lo`/`hi` draw the per-point
    min–max band (a ~10% wash) — the honest spread behind each median, from
    the adaptive repeats; the wash stays out of the legend, which the line
    layer owns. Log axes by default: sweep points are log-spaced, and a
    linear axis would pile four of five points into the left tenth of the
    panel.

    `overlay` (columns: hue, model, x, y) draws independent measurements as
    diamonds in the matching hue — e.g. the validation job's end-to-end
    numbers sitting on (or off) the curve the sweeps predict."""
    pts = pts.dropna(subset=[x, y])

    def _axis(values: pd.Series, log: bool) -> alt.Axis:
        ticks = _log_ticks(values) if log else None
        return alt.Axis(format="~s", values=ticks) if ticks else alt.Axis(format="~s")

    xs = alt.Scale(type="log", nice=False) if log_x else alt.Scale(zero=True)
    ys = alt.Scale(type="log", nice=False) if log_y else alt.Scale(zero=True)
    x_enc = alt.X(f"{x}:Q", title=x_title, scale=xs, axis=_axis(pts[x], log_x))
    y_enc = alt.Y(f"{y}:Q", title=y_title, scale=ys, axis=_axis(pts[y], log_y))
    scale = hue_scale if hue_scale is not None else alt.Undefined
    if hue:
        hue_enc = alt.Color(f"{hue}:N", title=None, scale=scale,
                            legend=alt.Legend(orient="bottom", columns=2,
                                              labelLimit=280))
        hue_quiet = alt.Color(f"{hue}:N", scale=scale, legend=None)
    else:
        hue_enc = hue_quiet = alt.value(LANE_SLOTS[0])

    # Chart only the columns the encodings use — passenger columns from the
    # loader (machine specs, …) would survive the concat below as object
    # dtype, where NaN dodges Altair's sanitization and breaks serialization.
    keep = [c for c in dict.fromkeys([hue, "model", x, y, lo, hi, "n_reps"])
            if c and c in pts.columns]
    pts = pts[keep].assign(is_overlay=False)
    has_overlay = overlay is not None and len(overlay)
    if has_overlay:
        over = overlay.rename(columns={"x": x, "y": y}).assign(is_overlay=True)
        pts = pd.concat([pts, over], ignore_index=True)
    base = alt.Chart(pts)
    curve = base.transform_filter("!datum.is_overlay")

    layers = []
    if lo and hi and {lo, hi} <= set(pts.columns):
        layers.append(curve.mark_area(opacity=0.12).encode(
            x=x_enc, y=alt.Y(f"{lo}:Q", scale=ys, title=y_title),
            y2=f"{hi}:Q", color=hue_quiet))
    # Points ride a 1.5px surface ring so they stay legible where lines cross.
    layers.append(curve.mark_line(
        strokeWidth=2,
        point=alt.OverlayMarkDef(filled=True, size=64,
                                 stroke=SURFACE, strokeWidth=1.5)).encode(
        x=x_enc, y=y_enc, color=hue_enc,
        tooltip=[c for c in (hue, "model", x, y, "n_reps") if c and c in pts.columns]))
    if has_overlay:
        layers.append(base.transform_filter("datum.is_overlay").mark_point(
            shape="diamond", filled=True, size=170,
            stroke=SURFACE, strokeWidth=1.5).encode(
            x=x_enc, y=y_enc, color=hue_quiet,
            tooltip=[c for c in (hue, "model", x, y) if c and c in pts.columns]))

    return _themed(_captioned(
        alt.layer(*layers).properties(width=width, height=230).facet(
            row=alt.Row("model:N", title=None,
                        header=alt.Header(labelOrient="top", labelAnchor="start",
                                          labelAngle=0, labelFontWeight="bold",
                                          labelFontSize=13))),
        caption))


def coverage(df: pd.DataFrame):
    """The raw record: status per submission/model/config × task, colour-coded
    with the shared status palette. Returns a pandas Styler. Unhealthy configs
    never ran a task, so they get a synthetic `(brain-check)` column rather than
    silently vanishing from the pivot."""
    df = df.copy()
    if df.task.isna().any():
        if isinstance(df.task.dtype, pd.CategoricalDtype):
            df["task"] = df.task.cat.add_categories(["(brain-check)"])
        df["task"] = df.task.fillna("(brain-check)")
    who = "submission" if "submission" in df else "machine"
    cov = df.pivot_table(index=[who, "model", "backend", "quant", "provider"],
                         columns="task", values="status", aggfunc="first",
                         observed=False)
    return cov.style.map(
        lambda v: f"background-color:{STATUS_COLORS.get(v, '#444')}; color:white"
        if isinstance(v, str) else ""
    )
