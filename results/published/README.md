# Published submissions

Shared, version-controlled benchmark runs — the baseline the analysis notebook
(`analysis/report.py`) reads by default. Everything else under `results/` is
local and gitignored; this folder is the exception.

## Submitting

1. Run the benchmark locally: `bench run --backend <key> --out results/<my-box>`.
2. Stage it as a submission:

   ```sh
   uv run --project harness bench publish results/<my-box> --name <my-box>
   ```

   This validates each `<backend>-results.json` against the contract, copies it
   and the matching `<backend>-raw.json.gz` into `results/published/<my-box>/`,
   and generates a `README.md` summarizing the spec (machine, sampling, and a
   `model × quant × provider` coverage table).
3. Regenerate the shared report so it includes the new submission:

   ```sh
   uv run --project analysis marimo export html --no-include-code \
     analysis/report.py -o results/published/report.html
   ```

4. Review the generated folder + report, then open a PR adding them.

## The report

`report.html` is the code-free static export of `analysis/report.py` over this
folder — every chart and number in it is computed from the submissions below
it, so it is regenerated (step 3) whenever one lands. Self-contained: open it
anywhere, no environment needed. On every push to `main` it is also served as
the repo's GitHub Pages site (`.github/workflows/pages.yml` — CI only
publishes the committed file, it never regenerates it).

## Layout

```
results/published/
  report.html           # static export of the analysis notebook over this folder
  <name>/
    README.md             # auto-generated spec summary (edit freely after)
    <backend>-results.json   # aggregated metrics — what the notebook loads
    <backend>-raw.json.gz    # raw per-spawn trace — re-aggregate with `bench aggregate`
```

The folder name (`<name>`) becomes the machine label in the notebook, so pick
something that identifies the box. Keep raw traces in: they let anyone re-derive
results if a metric definition changes, no re-inference needed.
