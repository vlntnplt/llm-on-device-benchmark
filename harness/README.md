# harness

The backend-agnostic Python tool. It enumerates work from the model registry
(`models.yaml`), spawns one backend process per `(model, variant, provider,
task)` cell, samples memory from the outside, and aggregates results. It never
reasons about which providers a machine has — it asks each exe.
(Why one process per cell, and the rest of the measurement model:
[ARCHITECTURE.md](../ARCHITECTURE.md).)

A [uv](https://docs.astral.sh/uv/) project:

```sh
uv sync
uv run bench --help

uv run bench fetch [model…] [--only ggml]                # pull artifacts into ../models
uv run bench check   --backend ggml --models ../models   # conformance-check a built exe
uv run bench plan    --backend ggml --models ../models   # enumerate cells, don't run
uv run bench run     --backend ggml --models ../models --out ../results
                     # per provider: ceiling probe; per cell: gate → sweep → job
uv run bench aggregate ../results/ggml-raw.json.gz       # re-derive results, no re-inference
uv run bench publish ../results/my-box --name my-box     # stage a submission
```

`run` takes `--iters K` (default 2), `--spawns S` (default 1), `--providers` to
restrict the sweep, and `--machine` to name the box in the results (default:
hostname). Progress goes to stderr; it writes `<backend>-raw.json.gz` (raw
per-spawn traces) and `<backend>-results.json` (aggregated `[p50, max]`) into
`--out`.

## The pipeline

1. **Enumerate** (`registry.py`) — `models.yaml` × fetched quants × each
   artifact's `providers` → the cell list.
2. **Invoke** (`commands/run.py`, `spawn.py`) — per `(model, variant,
   provider)`: one brain-check gate spawn; if healthy, the sweep, then the
   validation job (S spawns). Reads `backends/*/backend.toml` for the invoke
   command.
3. **Sample** (`sampling.py`) — on the job spawns only, a background thread
   reads `(wall_ns, rss, vram)` over the process tree every ~10 ms, never
   touching the measured process; `memory.py` aligns samples to event windows
   via the anchor.
4. **Aggregate** (`aggregate.py`, `metrics.py`) — pool to `[p50, max]` and
   write the results JSON.

Raw traces are persisted before any derivation, and `bench aggregate` re-runs
only the derivation through the same `aggregate.build` — changing how a metric
is summarized is a re-aggregate, not a multi-hour re-run. Samples store raw
measurements so every derivation stays re-aggregatable. The raw trace is
harness-internal, not part of the contract.

## Slow cells

Two guards keep a CPU-bound stack from grinding minutes per cell measuring
nothing new: the exe honours a soft `--deadline-ms` (always finishes iteration
1, stops before later ones once the deadline passes — every emitted iteration
is a whole decode), and the harness hard-kills at `--backstop-ms`. A bad
first job spawn is not re-ground. Unusable cells keep their status, never an
invented number — `too_slow` for genuine slowness (killed, or below the
~4 tok/s floor), `errored` for a spawn that died producing no sample
(crash/OOM/device-lost).

## Memory: RSS, on purpose

The backends run with mmap **off** — the shipped deployment configuration —
so weights are read into ordinary allocations at load and the whole CPU-side
footprint is resident: **RSS is the reported footprint** on every platform.
(File reads still warm the page cache, so the first-ever load of a model is
slower than a warm one — that difference is what `cold_start_ms` captures.)

On a GPU EP the weights leave RSS — where they go depends on the memory model:

- **NVIDIA** — real device VRAM, read per-PID via NVML, reported as its own
  pool. A Vulkan process appears in both NVML's compute and graphics lists with
  the same total, so VRAM is deduped per `(device, pid)` (max, not sum).
- **Linux integrated-AMD (Vulkan)** — unified memory, but llama.cpp copies the
  weights into **GTT** buffer objects the process never maps, so psutil RSS
  collapses to the CPU-side scaffolding (a 4 GB model: 670 MB RSS, 4.1 GB GTT —
  a 7× undercount). GTT is system RAM the kernel apertures for the GPU, so
  resident GTT (read off DRM fdinfo) is **folded into RSS**; the added GTT is
  disjoint from RSS, so there's no double-count. See `sampling.py:_drm_bytes`.

Phase figures: prefill is a ramp, so it reports its high-water mark; decode
reports **peak** (what must fit on the device) and **sustained** (median —
what generation occupies steady-state). They diverge when a transient rides
into the decode window, e.g. an EP compiling kernels at the first full-context
prefill. There is no isolated KV or weights figure — the phase footprints already
carry both (`ggml` preallocates KV with a clean plateau).

## Layout

```
bench/
  cli.py             entry point — argparse + dispatch only
  commands/          one module per subcommand: plan / run / aggregate / check / publish
  _log.py            log / warn / die → stderr (stdout stays clean)
  config.py          backend.toml → argv prefix; repo/models/tasks/results paths
  registry.py        models.yaml variants → `providers` → the cell list
  fetch.py           `bench fetch` — pull artifacts from the Hub per models.yaml
  tasks.py           load tasks, inline {document:}; gate vs timed role
  spawn.py           run one cell, sample the job spawns, validate events
  sampling.py        bg (wall_ns,rss,vram) sampler; NVML per-PID VRAM, DRM GTT
  metrics.py         events → ttft / prefill_tps / decode_tps / completion
  memory.py          align samples to event windows → phase footprints
  aggregate.py       pool S×K / S → [p50,max]; assemble the results runs
  machine.py         results.machine — host os / cpu / gpus
  schema.py          load + validate against the repo-root schema/
```

`metrics.py`, `memory.py`, and `aggregate.py` are pure (events/samples →
numbers), tested without spawning anything: `uv run pytest`. Lint/format with
`uv run ruff check bench` / `uv run ruff format bench`.
