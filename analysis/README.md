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
embedded as JSON and mounted by `assets/site.js`. The vega libraries are
fetched once at build time (versions pinned to the installed altair) and
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

## Layout: tested package, disposable notebook

- `bench_analysis/load.py` — results JSON → the tidy frame.
- `bench_analysis/prep.py` — derived views, pure pandas: config labels,
  hardware-descriptive machine names, phase splits for the stacked charts,
  the predicted-vs-measured memory model, status tallies, GPU-vs-CPU
  pairings.
- `bench_analysis/charts.py` — Altair builders sharing one visual language
  (green = measured winner/ok, fixed hues per status and backend). Imported
  only by the notebook, so the package's core import stays pandas-only.
- `bench_analysis/switcher.py` — kernel-free tab switches, as
  plain radios over pre-rendered panels. Takes rendered HTML rather than
  importing marimo, so the core import stays pandas-only.
- `report.py` — the marimo notebook: arranges prep's frames and charts'
  builders, writes the prose. Every number in the text is computed from the
  loaded frame, so it updates as submissions land. It presents measurements
  and highlights measured winners — conclusions are the reader's.
- `report.css` — the switcher styling the notebook injects into its export
  (`App(css_file=…)`).

`load.py`, `prep.py`, and `switcher.py` are the tested assets
(`uv run pytest`); the notebook is disposable. Lint with `uv run ruff check`.

## Switching in a static export

The HTML export has no kernel, so a `mo.ui` element cannot switch anything in
it — clicking one raises *"Static notebook: this notebook is not connected to a
kernel"*. Everything switchable is therefore pre-rendered and revealed by CSS:

- tab groups (`switcher.tabs`) — one panel per pre-rendered view.

The export also follows the reader's OS light/dark preference, from
`[tool.marimo.display] theme = "system"` in `pyproject.toml`. That setting is
what the export bakes in, and it has to be marimo's own: marimo picks the Vega
theme from it, so charts only get light-on-dark axes if marimo itself knows it
is dark. Page CSS cannot substitute — outputs render into a shadow root it
cannot reach. Charts are built on a transparent background so the page shows
through either way; mark and label colours are left exactly as computed, since
they carry meaning.
