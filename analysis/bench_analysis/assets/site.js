/* Tabs, model selectors, and chart mounting. Charts are vega-lite specs in
   <script type="application/json"> islands next to their mount points; they
   render lazily when their tab/panel first shows (vega sizes to 0 in a hidden
   container).

   Theming: specs are built with the light-mode palette; when the page is dark
   every colour string is swapped for its dark-surface step via the #dark-map
   island (charts.DARK_MAP), and axis/legend/grid ink is read off the page's
   own CSS tokens — one source of truth for chrome colours. A live scheme
   change re-mounts everything. */

const DARK_MQ = window.matchMedia("(prefers-color-scheme: dark)");
const DARK_MAP = JSON.parse(document.getElementById("dark-map").textContent);

function repaint(node) {
  /* Recursively swap light hexes for their dark steps inside a parsed spec. */
  if (Array.isArray(node)) return node.map(repaint);
  if (node && typeof node === "object") {
    const out = {};
    for (const k in node) out[k] = repaint(node[k]);
    return out;
  }
  return typeof node === "string" && DARK_MAP[node] !== undefined ? DARK_MAP[node] : node;
}

function chartConfig() {
  /* Chart chrome from the page's CSS tokens: recessive hairline grid,
     muted axis ink, readable legend ink. */
  const css = getComputedStyle(document.body);
  const v = (name) => css.getPropertyValue(name).trim();
  return {
    background: "transparent",
    font: "system-ui, sans-serif",
    axis: {
      labelColor: v("--muted"), titleColor: v("--muted"),
      gridColor: v("--hair"), domainColor: v("--border"), tickColor: v("--border"),
      labelFontSize: 11, titleFontSize: 12, titleFontWeight: 500,
    },
    legend: { labelColor: v("--fg2"), titleColor: v("--muted"), labelFontSize: 12 },
    title: { color: v("--muted") },
    header: { labelColor: v("--fg") },
    view: { stroke: v("--hair") },
  };
}

function mountCharts(root) {
  /* Only mount what is actually visible — a chart inside a hidden tab or a
     closed <details> would size to 0. Hidden ones mount later, when their
     container shows (tab switch, details toggle, select change). */
  for (const holder of root.querySelectorAll(".chart[data-spec]:not([data-mounted])")) {
    if (!visible(holder)) continue;
    holder.setAttribute("data-mounted", "1");
    let spec = JSON.parse(document.getElementById(holder.dataset.spec).textContent);
    if (DARK_MQ.matches) spec = repaint(spec);
    vegaEmbed(holder, spec, { actions: false, config: chartConfig() });
  }
}

function visible(el) {
  return el.offsetParent !== null;
}

/* Re-render every mounted chart when the colour scheme flips. */
DARK_MQ.addEventListener("change", () => {
  for (const holder of document.querySelectorAll(".chart[data-mounted]")) {
    holder.removeAttribute("data-mounted");
    holder.replaceChildren();
  }
  for (const panel of document.querySelectorAll("section[role=tabpanel]:not([hidden])"))
    mountCharts(panel);
});

/* Charts behind a <details> mount when it first opens. */
for (const d of document.querySelectorAll("details")) {
  d.addEventListener("toggle", () => {
    if (!d.open) return;
    mountCharts(d);
    for (const g of d.querySelectorAll("[data-group]")) syncGroup(g);
  });
}

/* Top-level tabs: hash-linkable (#models / #fleet / #evidence) and
   arrow-key navigable. */
for (const nav of document.querySelectorAll("nav.tabs")) {
  const buttons = [...nav.querySelectorAll("button")];
  const panels = buttons.map((b) => document.getElementById(b.dataset.panel));
  const names = buttons.map((b) => b.dataset.panel.replace(/^tab-/, ""));
  const show = (i, pushHash) => {
    buttons.forEach((b, j) => {
      b.setAttribute("aria-selected", i === j);
      b.tabIndex = i === j ? 0 : -1;
    });
    panels.forEach((p, j) => (p.hidden = i !== j));
    if (pushHash) history.replaceState(null, "", `#${names[i]}`);
    mountCharts(panels[i]);
    for (const g of panels[i].querySelectorAll("[data-group]")) syncGroup(g);
  };
  buttons.forEach((b, i) => b.addEventListener("click", () => show(i, true)));
  nav.addEventListener("keydown", (e) => {
    const cur = buttons.findIndex((b) => b.getAttribute("aria-selected") === "true");
    const to = { ArrowRight: cur + 1, ArrowLeft: cur - 1, Home: 0, End: buttons.length - 1 }[e.key];
    if (to === undefined) return;
    e.preventDefault();
    const i = (to + buttons.length) % buttons.length;
    show(i, true);
    buttons[i].focus();
  });
  const fromHash = () => {
    const i = names.indexOf(location.hash.replace(/^#/, ""));
    show(i >= 0 ? i : 0, false);
  };
  window.addEventListener("hashchange", fromHash);
  fromHash();
}

/* Model selectors: a <select data-group=g> shows the [data-group=g][data-key]
   block matching its value and hides its siblings. Selects sharing a
   data-sync key stay in step, so picking a model in one section follows the
   reader to the next. */
function syncGroup(container) {
  const sel = container.querySelector("select");
  if (!sel) return;
  for (const block of container.querySelectorAll("[data-key]")) {
    block.hidden = block.dataset.key !== sel.value;
    if (!block.hidden && visible(block)) mountCharts(block);
  }
}
for (const g of document.querySelectorAll("[data-group]")) {
  const sel = g.querySelector("select");
  if (sel)
    sel.addEventListener("change", () => {
      syncGroup(g);
      if (!sel.dataset.sync) return;
      for (const other of document.querySelectorAll(`select[data-sync="${sel.dataset.sync}"]`)) {
        if (other !== sel && [...other.options].some((o) => o.value === sel.value)) {
          other.value = sel.value;
          syncGroup(other.closest("[data-group]"));
        }
      }
    });
  syncGroup(g);
}
