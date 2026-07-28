# analysis — compare benchmark results across machines

A separate uv project from `harness/` on purpose: the harness ships to every
benchmark box and stays lean, while analysis pulls in pandas/altair/jinja and
runs only where you crunch numbers. It is a pure consumer of
`results.schema.json` — it reads what the harness writes and never touches the
contract.

```sh
uv sync

# build the report — one self-contained HTML (tabs, charts, the fleet
# calculator; no runtime dependencies, viewable offline):
uv run python -m bench_analysis.site   # → results/published/report.html
```

The site builder (`bench_analysis/site.py`) renders `templates/*.j2` over
context computed by the tested modules; charts are altair/vega-lite specs
embedded as JSON islands and mounted by `assets/site.js`. The vega libraries
are fetched once at build time (versions pinned to the installed altair) and
cached under `third_party/vega/` — nothing is fetched when the page is read.

## Results layout

Three loaders fan over `root/**/*-results.json`. Collect N machines into one
subdir per machine so filenames don't collide; the subdir name becomes the
machine label (flat files load too — labelled by the in-file `host`):

```
results/
  3090-box/   ggml-results.json
  m1-max/     ggml-results.json
```

- `load_results` — the validation job: one row per `(machine, backend, model,
  quant, provider)`, `[p50, max]` stats exploded into `<name>_p50` /
  `<name>_max`, geometry scalars as `geo_*`, machine memory config as
  `ram_*`. Gaps are visible, not absent: a null stat (VRAM on a CPU EP) is
  NaN, never 0, and a cell that produced no timing still gets a row, flagged
  by `status` (`ok` / `too_slow` / `errored` / `unhealthy`).
- `load_memory` — the memory cost curve: one row per allocator context point
  (`n_ctx`, pooled `weights_mb` / `kv_mb` / `compute_mb`) from the sweep's
  geometry. Exact allocator numbers — what an estimator fits, and what the
  report reads at the job's context against the sampled footprint.
- `load_sweeps` — one row per measured sweep point (prefill ms vs tokens,
  decode tok/s vs KV fill), each with its min–max spread and repeat count.
- `load_probes` — one row per device-ceiling point (GEMM TFLOP/s, copy GB/s).

Every loader pins the results `schema_version` — a file at another version is
a loud error, not a silently-misaligned frame.

## Layout: tested package, thin templates

- `bench_analysis/load.py` — results JSON → the tidy frame.
- `bench_analysis/prep.py` — derived views, pure pandas: config labels,
  hardware-descriptive machine names, phase splits for the stacked charts,
  the predicted-vs-measured memory model, status tallies, GPU-vs-CPU
  pairings.
- `bench_analysis/estimate.py` — the cost model: per-model work, per-lane
  affine fits, and the leave-one-out transfer error shown next to every
  prediction.
- `bench_analysis/charts.py` — Altair builders sharing one visual language
  (documented in its module docstring): validated categorical slots for lane
  identity, success-green reserved for `ok` outcomes, one status hue per
  failure mode, recessive greys for setup phases.
- `bench_analysis/site.py` — template context + rendering. Assigns each lane
  its colour slot (the same colour in every chart and in the Models tab's
  swatches), classifies lanes into hardware classes (CPU / integrated GPU /
  discrete GPU, from the bandwidth probes) for the Models table's ranges,
  computes the headline stats, and inlines everything.
- `templates/*.j2` — the three tabs (Models / Fleet / Evidence) and the page
  shell. Tabs are hash-linkable (`#models` / `#fleet` / `#evidence`).
  Models is the at-a-glance table — per-class ranges, no charts; Evidence
  holds the measurements behind it, every chart behind a `<details>`.
- `assets/site.css` — every colour on the page as custom properties, light
  and dark. `assets/site.js` mounts charts lazily, feeds them the page's own
  ink tokens, and swaps series colours for their dark-surface steps (the
  `#dark-map` island from `charts.DARK_MAP`) when the reader is dark.
  `assets/fleet.js` is the fleet calculator; its tier ramp is CSS classes so
  the colours live with every other token.

`load.py`, `prep.py`, `estimate.py`, `charts.py`, and `site.py` are the
tested assets (`uv run pytest`); templates and assets are exercised through
`site.build` in `tests/test_site.py`. Lint with `uv run ruff check`.

## Theming in a static export

The page follows the reader's OS light/dark preference with no rebuild:
chrome colours are CSS custom properties, and charts render on transparent
backgrounds with their axis/legend ink read from those same properties at
mount time. Series colours can't come from CSS (vega specs carry hex), so
specs are built with the light palette and `site.js` rewrites each colour to
its validated dark step when mounting on a dark page — both palettes are
chosen for their surface rather than one compromise holding up on both.
