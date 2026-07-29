/* The fleet calculator. Coefficients arrive in #fleet-data (JSON): per-model
   costs, per-class pooled (t0, eta), and the leave-one-out error that must be
   shown next to every prediction. The math mirrors bench_analysis.estimate:
   time = t0 + work / (eta × ceiling).

   Tier colours are CSS classes (t0/t1/t2/tslow/tnofit), so the ordinal ramp
   and its light/dark steps live in site.css with every other token. Every
   coverage bar gets a text readout of its mix — the segments echo it, they
   are never the only way to read the number — and each model row expands to
   the per-cohort predictions behind its bar. */

const DATA = JSON.parse(document.getElementById("fleet-data").textContent);

const COHORTS = [
  ["iGPU · low", 0, 8, 30, 1.5],
  ["iGPU · mid", 40, 16, 60, 3.5],
  ["iGPU · high", 0, 32, 120, 8],
  ["discrete · low", 10, 6, 200, 10],
  ["discrete · mid", 20, 12, 400, 40],
  ["discrete · high", 5, 16, 800, 130],
  ["Apple · low", 0, 8, 68, 2.5],
  ["Apple · mid", 15, 24, 270, 7],
  ["Apple · high", 0, 36, 540, 16],
];
const TIERS = [["instant", 1, 20], ["usable", 3, 8], ["patient", 10, 3]];
const TIER_CLASSES = ["t0", "t1", "t2", "tslow", "tnofit"];

const open = new Set(); // model rows whose cohort detail stays expanded

function el(tag, attrs, text) {
  const e = document.createElement(tag);
  for (const k in attrs || {}) e.setAttribute(k, attrs[k]);
  if (text !== undefined) e.textContent = text;
  return e;
}
const val = (id) => parseFloat(document.getElementById(id).value) || 0;
const cls = (kind) =>
  DATA.classes.find((c) => c.lane_class === "gpu" && c.kind === kind) ||
  { t0_ms: 0, eta: 1 };
const fmtS = (s) => (s < 10 ? s.toFixed(1) : s.toFixed(0)) + " s";
const fmtTps = (t) => (t < 10 ? t.toFixed(1) : t.toFixed(0));

function buildInputs() {
  const tb = document.querySelector("#fleet-cohorts tbody");
  COHORTS.forEach((c, i) => {
    const tr = el("tr");
    tr.appendChild(el("td", {}, c[0]));
    const count = el("td");
    count.appendChild(el("input", { type: "number", id: `c${i}-1`, value: c[1], step: "any" }));
    tr.appendChild(count);
    tr.appendChild(el("td", { class: "preset", id: `c${i}-preset` }));
    for (let j = 2; j <= 4; j++) {
      const td = el("td", { class: "adv" });
      td.appendChild(el("input", { type: "number", id: `c${i}-${j}`, value: c[j], step: "any" }));
      tr.appendChild(td);
    }
    tb.appendChild(tr);
  });
  const adv = document.getElementById("fleet-advanced");
  adv.addEventListener("change", () =>
    document.getElementById("fleet-root").classList.toggle("advanced", adv.checked));
  const tt = document.querySelector("#fleet-tiers tbody");
  TIERS.forEach((t, i) => {
    const tr = el("tr");
    const name = el("td");
    name.appendChild(el("input", { id: `t${i}-name`, class: "name", value: t[0] }));
    tr.appendChild(name);
    for (let j = 1; j <= 2; j++) {
      const td = el("td");
      td.appendChild(el("input", { type: "number", id: `t${i}-${j}`, value: t[j], step: "any" }));
      tr.appendChild(td);
    }
    tt.appendChild(tr);
  });
  document.getElementById("fleet-root").addEventListener("input", render);
}

function predict(m, mem, bw, tf, prompt, out) {
  const ctx = prompt + out;
  const kvGb = (m.kv_state_mb + (m.kv_mb_per_1k * ctx) / 1024) / 1024;
  const need = m.file_gb + kvGb + 1.0;
  if (need > mem) return { fits: false, need };
  const d = cls("decode"), p = cls("prefill");
  const tps = 1 / (d.t0_ms / 1e3 + (m.body_gb + kvGb) / (d.eta * bw));
  const ttft =
    (Math.ceil(prompt / 512) * p.t0_ms) / 1e3 +
    (2 * m.body_gparams * prompt) / 1e3 / (p.eta * tf);
  return { fits: true, need, tps, ttft };
}

function render() {
  COHORTS.forEach((_, i) => {
    document.getElementById(`c${i}-preset`).textContent =
      `${val(`c${i}-2`)} GB · ${val(`c${i}-3`)} GB/s · ${val(`c${i}-4`)} TF`;
  });
  const prompt = val("w-prompt"), out = val("w-out");
  const tiers = TIERS.map((t, i) => ({
    name: document.getElementById(`t${i}-name`).value,
    ttft: val(`t${i}-1`),
    tps: val(`t${i}-2`),
  })).sort((a, b) => a.ttft - b.ttft);
  const labels = [...tiers.map((t) => t.name), "too slow", "doesn't fit"];

  const legend = document.getElementById("fleet-legend");
  legend.replaceChildren(
    ...labels.map((l, i) => {
      const s = el("span");
      s.appendChild(el("i", { class: TIER_CLASSES[i] }));
      s.appendChild(document.createTextNode(l));
      return s;
    })
  );

  const tb = document.querySelector("#fleet-results tbody");
  tb.replaceChildren();
  const total = COHORTS.reduce((s, _, i) => s + val(`c${i}-1`), 0);
  for (const m of DATA.models) {
    const key = `${m.model} ${m.quant}`;
    const buckets = new Array(tiers.length + 2).fill(0);
    const perCohort = [];
    COHORTS.forEach((c, i) => {
      const n = val(`c${i}-1`);
      if (!n) return;
      const r = predict(m, val(`c${i}-2`), val(`c${i}-3`), val(`c${i}-4`), prompt, out);
      let bucket = tiers.length + 1;
      if (r.fits) {
        const t = tiers.findIndex((t) => r.ttft <= t.ttft && r.tps >= t.tps);
        bucket = t >= 0 ? t : tiers.length;
      }
      buckets[bucket] += n;
      perCohort.push({ name: c[0], n, r, outcome: labels[bucket] });
    });

    const tr = el("tr");
    const nameTd = el("td");
    const btn = el("button", {
      class: "disclose",
      "aria-expanded": open.has(key) ? "true" : "false",
    }, key);
    btn.addEventListener("click", () => {
      open.has(key) ? open.delete(key) : open.add(key);
      render();
    });
    nameTd.appendChild(btn);
    tr.appendChild(nameTd);

    const bar = el("td");
    const b = el("div", { class: "bar" });
    buckets.forEach((n, i) => {
      if (n <= 0) return;
      const pct = (100 * n) / total;
      const seg = el("div", {
        class: `seg ${TIER_CLASSES[i]}`,
        style: `width:${pct}%`,
        title: `${labels[i]}: ${pct.toFixed(0)}%`,
      });
      if (pct >= 12) seg.textContent = `${pct.toFixed(0)}%`;
      b.appendChild(seg);
    });
    bar.appendChild(b);
    tr.appendChild(bar);

    const served = buckets.slice(0, tiers.length).reduce((a, x) => a + x, 0);
    const mixTd = el("td", { class: "mix" });
    if (total) {
      mixTd.appendChild(el("strong", {}, `${((100 * served) / total).toFixed(0)}% served`));
      const parts = buckets
        .map((n, i) => (n > 0 ? `${((100 * n) / total).toFixed(0)}% ${labels[i]}` : null))
        .filter(Boolean);
      mixTd.appendChild(el("span", { class: "muted" }, ` — ${parts.join(" · ")}`));
    } else {
      mixTd.textContent = "—";
    }
    tr.appendChild(mixTd);
    tb.appendChild(tr);

    if (open.has(key)) {
      const dtr = el("tr", { class: "detail" });
      const td = el("td", { colspan: "3" });
      const t = el("table");
      const head = el("tr");
      for (const h of ["cohort", "count", "needs", "first token", "tok/s", "outcome"])
        head.appendChild(el("th", {}, h));
      t.appendChild(el("thead")).appendChild(head);
      const body = t.appendChild(el("tbody"));
      for (const pc of perCohort) {
        const row = el("tr");
        row.appendChild(el("td", {}, pc.name));
        row.appendChild(el("td", {}, `${pc.n}`));
        row.appendChild(el("td", {}, `${pc.r.need.toFixed(1)} GB`));
        row.appendChild(el("td", {}, pc.r.fits ? fmtS(pc.r.ttft) : "—"));
        row.appendChild(el("td", {}, pc.r.fits ? fmtTps(pc.r.tps) : "—"));
        row.appendChild(el("td", {}, pc.outcome));
        body.appendChild(row);
      }
      td.appendChild(t);
      dtr.appendChild(td);
      tb.appendChild(dtr);
    }
  }

  const parts = DATA.loo.map(
    (l) => `${l.lane_class} ${l.kind} ±${(100 * l.median_err).toFixed(0)}%`
  );
  document.getElementById("fleet-honesty").textContent =
    "Model error (leave-one-out across the measured machines): " +
    parts.join(" · ") +
    ". Boundary calls within those margins can go either way.";
}

buildInputs();
render();
