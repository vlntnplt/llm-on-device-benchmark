# On-device LLM inference benchmark

A simple, trustable harness to measure the cost of on-device LLM inference
— [llama.cpp](https://github.com/ggml-org/llama.cpp) (`ggml`) —
across machines: per provider a bare **device ceiling probe**, per model the
runtime-reported **geometry**, **prefill/decode sweeps** to 8k context, and one
real **validation job** measuring load time, time-to-first-token, decode
throughput, and memory end to end.

**[ARCHITECTURE.md](ARCHITECTURE.md)** is the map — components, the contract,
the measurement model. This file is how to run the thing. Each component's
README has its specifics.

## Layout

```
ARCHITECTURE.md    the map — read this first
schema/            JSON Schemas for the events + results objects — the wire contract
tasks/             task catalog, corpora, brain-check gate
models.yaml        model registry + fetch spec (per model: gguf repo + quants)
backends/
  ggml/              llama.cpp backend (C++) — builds build/bench-ggml
harness/            Python tool (uv) — bench fetch/check/plan/run/aggregate/publish
results/           harness output; local & gitignored except published/ submissions
analysis/          Python project (uv) — cross-machine comparison notebook
```

Untracked, you provide: `models/` (artifacts pulled by `bench fetch`) and
optionally `third_party/` (local stack checkouts for offline builds or hacking
— each backend otherwise fetches its own pinned stack at build time).

## Run a benchmark

1. **Build the backend** — an independent unit with its own README and
   toolchain ([backends/ggml](backends/ggml/README.md)). The harness
   skips a backend whose exe isn't runnable.

2. **Fetch models and run** — the harness is a [uv](https://docs.astral.sh/uv/)
   project:

   ```sh
   cd harness
   uv sync

   uv run bench fetch gemma4-E2B                          # pull artifacts into ../models
   uv run bench check --backend ggml --models ../models   # conformance-check a built exe
   uv run bench plan  --backend ggml --models ../models   # enumerate cells, don't run
   uv run bench run   --backend ggml --models ../models --tasks ../tasks \
                      --out ../results --machine my-box
   ```

   `run` writes two files per backend: `<backend>-raw.json.gz` (raw per-spawn
   traces) and `<backend>-results.json` (aggregated `[p50, max]`).
   `bench aggregate` recomputes the second from the first — no re-inference.
   `--machine` names the box in the results (default: hostname).

3. **Compare** — load one or many machines' results with the separate
   [`analysis/`](analysis/README.md) project; share a run via
   [`bench publish`](results/published/README.md). The report over the
   published submissions (`results/published/report.html`) is served as the
   repo's GitHub Pages site.

## Adding a backend

A backend is one directory under `backends/` that builds to an exe, implements
the `providers` / `run` / `version` CLI, emits schema-valid events on stdout,
and registers itself with a `backend.toml`:

```toml
key  = "ggml"                        # must match the events object's `backend`
name = "llama.cpp / GGML"            # human label
cmd  = ["{dir}/build/bench-ggml"]    # argv prefix; harness substitutes {dir}, appends the subcommand
```

The `key` also selects the backend's `models.yaml` block (`ggml`→`gguf`);
the directory name is free. The contract rules are summarized in
[ARCHITECTURE.md](ARCHITECTURE.md) and [CLAUDE.md](CLAUDE.md); the schemas in
[`schema/`](schema/) are the field-level truth. `bench check` is the
conformance gate.

## Reproducibility

Every events object embeds `<exe> version` — exact library commits, build
flags, `use_mmap`, thread count. A quant label names stack-specific math
(`q4` = Q4_K_M), so results are compared within a label, and each result's
stack versions qualify it.
