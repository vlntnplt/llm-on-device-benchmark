# Published submissions

Shared, version-controlled benchmark runs — the baseline the report builder
(`bench_analysis.site`) reads by default. Everything else under `results/` is
local and gitignored; this folder is the exception.

A submission measures a machine's cost function: per provider a device ceiling
probe (GEMM, copy bandwidth), and per (model, quant, provider) the model
geometry as the runtime reports it, prefill/decode sweeps to 8k context, and
one real validation job. Run with `sudo` where possible so the installed
memory config (dmidecode) lands in the machine block — it is the source of the
machine's nominal bandwidth.

## Submitting

1. Run the benchmark locally: `bench run --backend <key> --out results/<my-box>`.
2. Stage it as a submission:

   ```sh
   uv run --project harness bench publish results/<my-box> --name <my-box>
   ```

   This validates each `<backend>-results.json` against the contract, copies it
   and the matching `<backend>-raw.json.gz` into `results/published/<my-box>/`,
   and generates a `README.md` summarizing the spec (machine incl. memory
   config, ceiling probes, and a `model × quant × provider` sweep/job coverage
   table).
3. Regenerate the shared report so it includes the new submission:

   ```sh
   uv run --project analysis python -m bench_analysis.site
   ```

4. Review the generated folder + report, then open a PR adding them.

## The report

`report.html` is built by `bench_analysis.site` over this folder — every
chart and number in it is computed from the submissions below it, so it is
regenerated (step 3) whenever one lands. Self-contained: open it anywhere, no
environment needed, nothing fetched at view time. On every push to `main` it is also served as
the repo's GitHub Pages site (`.github/workflows/pages.yml` — CI only
publishes the committed file, it never regenerates it).

## Layout

```
results/published/
  report.html           # built by bench_analysis.site over this folder
  <name>/
    README.md             # auto-generated spec summary (edit freely after)
    <backend>-results.json   # aggregated metrics — what the report loads
    <backend>-raw.json.gz    # raw per-spawn trace — re-aggregate with `bench aggregate`
```

The folder name (`<name>`) becomes the submission label in the report, so pick
something that identifies the box. Keep raw traces in: they let anyone re-derive
results if a metric definition changes, no re-inference needed.
