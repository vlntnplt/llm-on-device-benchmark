/* Tabs, model selectors, and chart mounting. Charts are vega-lite specs in
   <script type="application/json"> islands next to their mount points; they
   render lazily when their tab/panel first shows (vega sizes to 0 in a hidden
   container). Theme follows prefers-color-scheme. */

const DARK = window.matchMedia("(prefers-color-scheme: dark)").matches;

function mountCharts(root) {
  for (const holder of root.querySelectorAll(".chart[data-spec]:not([data-mounted])")) {
    holder.setAttribute("data-mounted", "1");
    const spec = JSON.parse(document.getElementById(holder.dataset.spec).textContent);
    vegaEmbed(holder, spec, {
      actions: false,
      theme: DARK ? "dark" : undefined,
      config: { background: "transparent" },
    });
  }
}

function visible(el) {
  return el.offsetParent !== null;
}

/* Top-level tabs. */
for (const nav of document.querySelectorAll("nav.tabs")) {
  const buttons = [...nav.querySelectorAll("button")];
  const panels = buttons.map((b) => document.getElementById(b.dataset.panel));
  const show = (i) => {
    buttons.forEach((b, j) => b.setAttribute("aria-selected", i === j));
    panels.forEach((p, j) => (p.hidden = i !== j));
    mountCharts(panels[i]);
    for (const g of panels[i].querySelectorAll("[data-group]")) syncGroup(g);
  };
  buttons.forEach((b, i) => b.addEventListener("click", () => show(i)));
  show(0);
}

/* Model selectors: a <select data-group=g> shows the [data-group=g][data-key]
   block matching its value and hides its siblings. */
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
  if (sel) sel.addEventListener("change", () => syncGroup(g));
  syncGroup(g);
}
