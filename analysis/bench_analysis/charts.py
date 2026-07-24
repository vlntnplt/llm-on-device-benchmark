"""Altair builders for the comparison notebook.

One place for the visual language, so every chart speaks it:

- **green** marks a measured winner or an `ok` outcome — never a data series.
- failure modes keep one hue everywhere (markers, bars, coverage):
  amber `too_slow`, purple `errored`, dark red `unhealthy`.
- backends keep one hue wherever they're contrasted: teal ggml, orange tjs.
  Teal doubles as "accelerated" and slate as "cpu" in the GPU-vs-CPU charts.
- in phase stacks, pale slates = setup, vivid indigo/sky = the generation work,
  so the eye lands on the part that answers the user.
- reference rules (the 1 s responsiveness line) are muted gray and dashed —
  annotations, not data.

Builders take the frames `prep` derives and return Altair charts; the notebook
just composes them. Only the notebook imports this module (altair lives in the
`notebook` dependency group), so the package's core import stays pandas-only.
"""

from __future__ import annotations

import altair as alt
import pandas as pd

from . import prep

ACCENT = "#1e8e3e"   # the only green anywhere: a measured win / ok
MUTED = "#8a939b"    # secondary labels, reference rules

BACKEND_COLORS = {"ggml": "#2a9d8f", "tjs": "#e8913a"}
STATUS_COLORS = {"ok": ACCENT, "too_slow": "#b45309",
                 "errored": "#7d3c98", "unhealthy": "#7f1d1d"}

# Setup recedes (pale slates), work pops (indigo prefill, sky decode).
TIME_COLORS = dict(zip(prep.TIME_PHASES,
                       ["#94a3b8", "#b9c5d1", "#dde3e9", "#5e60ce", "#0ea5e9"],
                       strict=True))
# Same family logic: slate = host RAM, sky = device VRAM, pale = transient.
MEMORY_COLORS = dict(zip(prep.MEMORY_PHASES,
                         ["#64748b", "#0ea5e9", "#dde3e9"], strict=True))
CPU_GPU_COLORS = {"gpu": "#2a9d8f", "cpu": "#64748b"}

_DNF_COLORS = {v: STATUS_COLORS[k] for k, v in prep.FAIL_LABELS.items()}


def _themed(chart: alt.Chart) -> alt.Chart:
    """Let the page's background show through, so one rendering reads on both a
    light and a dark page. Vega's default opaque white would otherwise punch a
    panel out of a dark report; the text colours are retargeted in report.css."""
    return chart.configure(background="transparent")


def _captioned(chart: alt.Chart, text: str) -> alt.Chart:
    """One caption along the bottom of the whole chart — a per-facet axis title
    would repeat (and collide) under every facet column."""
    return chart.properties(
        title=alt.TitleParams(text, orient="bottom", anchor="middle",
                              fontWeight="normal", fontSize=12, dy=6))


def stacked(g: pd.DataFrame, colors: dict[str, str], caption: str, *,
            width: int = 430, fmt: str = ".0f",
            dnf: pd.DataFrame | None = None,
            config_order: list[str] | None = None) -> alt.Chart:
    """Horizontal stacked bars, one row facet per model, ranked by total
    ascending. Row facets (not columns) so the chart never outgrows the page —
    a report scrolls down, not sideways.

    `g` is long: one row per (model, config, phase) with a numeric `value`.
    `colors` maps phase → fill; its dict order is the bottom→top draw order.
    The total per config is labelled at the bar end — green for the winner per
    model. `dnf` (from `prep.failures`) adds the configs with no usable sample
    as full-width markers sorted to the bottom, coloured by failure mode.
    `config_order` (e.g. from `prep.shared_config_order`) pins the row order —
    pass the same list to sibling charts (tabs) so configs don't jump rows.
    """
    order = list(colors)
    g = g[g.phase.isin(order)].copy()
    g["rank"] = g.phase.map({p: i for i, p in enumerate(order)})

    tot = g.groupby(["model", "config"], observed=True)["value"].sum().reset_index(name="total")
    tot["best"] = tot.groupby("model")["total"].transform("min") == tot["total"]
    g = g.merge(tot, on=["model", "config"])

    # Fastest/lightest first, shared across facets so rows line up; no-sample
    # configs sort to the bottom (worst). Configs the pinned order doesn't know
    # (e.g. failed in every task) append at the end.
    cfg_order = (list(config_order) if config_order is not None
                 else tot.groupby("config")["total"].min().sort_values().index.tolist())
    has_dnf = dnf is not None and len(dnf)
    if has_dnf:
        cfg_order += [c for c in dnf.config.unique() if c not in cfg_order]

    # A faceted layer chart must share ONE top-level dataset — Vega-Lite rejects
    # per-layer data under a facet. So stack the bar rows and the no-sample rows
    # together and split them back per layer with a filter on `is_dnf`.
    g["is_dnf"] = False
    frames = [g]
    if has_dnf:
        d = dnf.copy()
        d["x"] = float(tot["total"].max()) if len(tot) else 1.0
        d["mid"] = d["x"] / 2
        d["is_dnf"] = True
        frames.append(d)
    data = pd.concat(frames, ignore_index=True)

    # Hardware-named configs run long ("Ryzen 9 9950X + RTX 5080 · ggml-vulkan");
    # the default ~180px label limit would clip them.
    y = alt.Y("config:N", title=None, sort=cfg_order,
              axis=alt.Axis(labelLimit=260))
    base = alt.Chart(data)
    bars = base.transform_filter("!datum.is_dnf").mark_bar().encode(
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
        f"!datum.is_dnf && datum.phase == '{order[0]}'"
    ).mark_text(align="left", dx=4, fontWeight="bold").encode(
        y, x=alt.X("total:Q"), text=alt.Text("total:Q", format=fmt),
        color=alt.condition("datum.best", alt.value(ACCENT), alt.value(MUTED)),
    )
    layers = [bars, labels]

    if has_dnf:
        dom = list(_DNF_COLORS)
        cscale = alt.Scale(domain=dom, range=[_DNF_COLORS[k] for k in dom])
        dbase = base.transform_filter("datum.is_dnf")
        layers += [
            dbase.mark_bar(opacity=0.22).encode(
                y, x=alt.X("x:Q", title=None),
                color=alt.Color("label:N", scale=cscale, legend=None),
                tooltip=["model", "config", alt.Tooltip("label:N", title="status")],
            ),
            dbase.mark_text(fontWeight="bold").encode(
                y, x=alt.X("mid:Q"), text=alt.Text("label:N"),
                color=alt.Color("label:N", scale=cscale, legend=None),
            ),
        ]

    # Step-sized rows: each facet is exactly as tall as the configs it holds.
    # Model names ride above each panel rather than in a rotated left gutter.
    return _themed(_captioned(
        alt.layer(*layers).properties(width=width, height=alt.Step(20)).facet(
            row=alt.Row("model:N", title=None,
                        header=alt.Header(labelOrient="top", labelAnchor="start",
                                          labelAngle=0, labelFontWeight="bold",
                                          labelFontSize=13))),
        caption))


def status_bars(cells: pd.DataFrame) -> alt.Chart:
    """Outcome counts per backend·provider (from `prep.status_cells`), stacked by
    status, with an ok/total label — green only when the row is clean."""
    order = [s for s in STATUS_COLORS if s in set(cells.status)]
    who_order = sorted(cells.who.unique())

    tot = cells.groupby("who", observed=True).n.sum()
    n_ok = (cells[cells.status == "ok"].groupby("who", observed=True).n.sum()
            .reindex(tot.index, fill_value=0))
    lab = pd.DataFrame({"who": tot.index, "total": tot.to_numpy(), "n_ok": n_ok.to_numpy()})
    lab["lbl"] = lab.n_ok.astype(str) + "/" + lab.total.astype(str)
    lab["clean"] = lab.n_ok == lab.total

    bars = alt.Chart(cells).mark_bar().encode(
        y=alt.Y("who:N", title=None, sort=who_order),
        x=alt.X("n:Q", title="cells attempted"),
        color=alt.Color("status:N", title=None, sort=order,
                        scale=alt.Scale(domain=order, range=[STATUS_COLORS[s] for s in order])),
        tooltip=["who", "status", "n"],
    )
    labels = alt.Chart(lab).mark_text(align="left", dx=4, fontWeight="bold").encode(
        y=alt.Y("who:N", sort=who_order), x="total:Q", text="lbl:N",
        color=alt.condition("datum.clean", alt.value(ACCENT), alt.value(STATUS_COLORS["too_slow"])),
    )
    return _themed((bars + labels).properties(width=420, height=24 * len(who_order)))


def dumbbell(best: pd.DataFrame, task_order: list[str], caption: str, *,
             row: str = "lane", value: str = "total_s",
             value_title: str = "total (s)", width: int = 125,
             y: str = "model", y_sort: list[str] | None = None,
             hue: str = "backend", colors: dict[str, str] | None = None,
             ref_x: float | None = None) -> alt.Chart:
    """A head-to-head (from a `prep` matchup frame): one dot per `hue` value on
    a log axis, faceted `row` × task, the gap labelled ×heavy/light. Cells
    where only one side produced a sample get no gap label — that's a walkover,
    not a tie. `ref_x` draws a dashed reference rule (e.g. the 1 s line)."""
    colors = colors or BACKEND_COLORS
    dom = sorted(best[hue].unique())
    y_enc = alt.Y(f"{y}:N", title=None, sort=y_sort)
    base = alt.Chart(best).transform_calculate(gap="'×' + format(datum.ratio, '.1f')")

    rule = base.mark_rule(color="#cbd2d9", strokeWidth=2).encode(
        y=y_enc, x="lo:Q", x2="hi:Q")
    pts = base.mark_point(filled=True, size=90).encode(
        y=y_enc,
        x=alt.X(f"{value}:Q", title=None,
                scale=alt.Scale(type="log", nice=False, padding=12),
                axis=alt.Axis(format="~r")),
        color=alt.Color(f"{hue}:N", title=None,
                        scale=alt.Scale(domain=dom,
                                        range=[colors.get(b, MUTED) for b in dom]),
                        legend=alt.Legend(orient="bottom")),
        tooltip=[*(c for c in dict.fromkeys([row, "machine", "model", "task", hue,
                                             "provider", "quant"]) if c in best.columns),
                 alt.Tooltip(f"{value}:Q", title=value_title, format=".1f")],
    )
    gap = base.transform_filter(
        f"datum.n_backends >= 2 && datum.{value} >= datum.hi"
    ).mark_text(align="left", dx=9, color=MUTED, fontWeight="bold").encode(
        y=y_enc, x=f"{value}:Q", text="gap:N")
    layers = [rule, pts, gap]
    if ref_x is not None:
        layers.append(alt.Chart(best).mark_rule(color=MUTED, strokeDash=[4, 3])
                      .encode(x=alt.datum(ref_x)))

    # Row facet labels are rotated 90° by default — silicon names overlap.
    # Lay them flat; the wider left gutter is worth it. The right padding is
    # headroom for the last column's gap labels, which hang past the hi dot.
    return _themed(_captioned(
        alt.layer(*layers).properties(width=width, height=alt.Step(26)).facet(
            row=alt.Row(f"{row}:N", title=None,
                        header=alt.Header(labelAngle=0, labelAlign="left", labelLimit=230)),
            column=alt.Column("task:N", title=None, sort=task_order)),
        caption).properties(padding={"left": 5, "top": 5, "bottom": 5, "right": 42}))


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
