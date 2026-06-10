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
3. Review the generated folder, then open a PR adding it.

## Layout

```
results/published/
  <name>/
    README.md             # auto-generated spec summary (edit freely after)
    <backend>-results.json   # aggregated metrics — what the notebook loads
    <backend>-raw.json.gz    # raw per-spawn trace — re-aggregate with `bench aggregate`
```

The folder name (`<name>`) becomes the machine label in the notebook, so pick
something that identifies the box. Keep raw traces in: they let anyone re-derive
results if a metric definition changes, no re-inference needed.
