# analysis — compare benchmark results across machines

A separate uv project from `harness/` on purpose: the harness ships to every
benchmark box and stays lean, while analysis pulls in pandas (and optionally
marimo/altair) and runs only where you crunch numbers. It is a pure consumer of
`results.schema.json` — it reads what the harness writes and never touches the
contract.

```sh
uv sync                        # just the loader + prep (pandas)
uv sync --group notebook       # + marimo, altair, jinja2

# explore + compare (report.py is a marimo notebook — a plain .py file):
uv run marimo edit report.py

# a static snapshot to hand off (no env needed to read it);
# --no-include-code makes it read as a report rather than a notebook:
uv run marimo export html --no-include-code report.py -o report.html
```

## Results layout

`load_results(root)` fans over `root/**/*-results.json`. Collect N machines
into one subdir per machine so filenames don't collide; the subdir name becomes
the machine label (flat files load too — labelled by the in-file `host`):

```
results/
  3090-box/   ggml-results.json  tjs-results.json
  m1-max/     ggml-results.json  tjs-results.json
```

The frame is long/tidy: one row per `(machine, backend, model, quant, provider,
device, task)`; every `[p50, max]` stat explodes into `<name>_p50` /
`<name>_max`. Gaps are visible, not absent: a null stat (VRAM on a CPU EP) is
NaN, never 0, and a cell that produced no timing still gets a row, flagged by
`status` (`ok` / `too_slow` / `errored` / `unhealthy`). The loader pins the
results `schema_version` — a file at another version is a loud error, not a
silently-misaligned frame.

## Layout: tested package, disposable notebook

- `bench_analysis/load.py` — results JSON → the tidy frame.
- `bench_analysis/prep.py` — derived views, pure pandas: task ordering, config
  labels, hardware-descriptive machine names, phase splits for the stacked
  charts, per-backend status tallies, head-to-head fastest configs, GPU-vs-CPU
  pairings.
- `bench_analysis/charts.py` — Altair builders sharing one visual language
  (green = measured winner/ok, fixed hues per status and backend). Imported
  only by the notebook, so the package's core import stays pandas-only.
- `bench_analysis/switcher.py` — the report's task and backend switches, as
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

- **context size** — one panel per task, per chart.
- **backend** — a page-level *both backends / ggml only* switch. tjs sets the
  scale in §1 and §2, compressing the ggml configs against the axis; dropping it
  makes the remaining spread readable. Both states are rendered in Python
  because row order, axis scale, and the per-model winner are computed over the
  rows a chart shows. §4 and §5 ignore the switch — they exist to compare the
  two backends.

The export also follows the reader's OS light/dark preference, from
`[tool.marimo.display] theme = "system"` in `pyproject.toml`. That setting is
what the export bakes in, and it has to be marimo's own: marimo picks the Vega
theme from it, so charts only get light-on-dark axes if marimo itself knows it
is dark. Page CSS cannot substitute — outputs render into a shadow root it
cannot reach. Charts are built on a transparent background so the page shows
through either way; mark and label colours are left exactly as computed, since
they carry meaning.
