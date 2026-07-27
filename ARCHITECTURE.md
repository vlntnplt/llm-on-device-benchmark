# Architecture

A harness to measure the cost of on-device inference across machines — not
just what a task took, but the **parameters of the cost function**: what the
silicon can do (ceiling probes), what the model is (runtime-reported
geometry), how cost scales (prefill/decode sweeps to 8k context), and whether
those parameters reproduce a real workload (one validation job).

| backend | stack | language | artifact |
|---------|-------|----------|----------|
| `ggml`  | [llama.cpp](https://github.com/ggml-org/llama.cpp) | C++ (native) | one `.gguf`, runs on any provider |

## Components

```
schema/            the wire contract — events (backend → harness) and results
                   (harness → analysis) JSON Schemas
tasks/             what the benchmark runs — timed catalog, corpora, the
                   brain-check provider-health gate
models.yaml        what it runs on — model registry + Hub fetch spec
backends/          one independently-buildable exe per stack (ggml)
harness/            backend-agnostic Python tool — enumerates cells, spawns
                   backends, samples memory, aggregates, publishes
analysis/          separate Python project — loads results into pandas,
                   compares across machines (marimo notebook)
results/published/ version-controlled benchmark submissions
```

Each component has a README with its specifics; this file is the map.

## The protocol

Per provider, once: a **probe** — bare f16 GEMM and buffer-copy throughput on
the exact device inference selects, no model loaded. The ceilings every model
number is later divided by.

Per `(model, variant, provider)` cell, in order:

1. **Gate** — the brain-check: one spawn running a trivial three-turn task,
   every turn expect-checked. Any miss → the cell is `unhealthy`, nothing
   else runs.
2. **Sweep** — the cost function in one instrumented pass, synthetic tokens,
   no chat semantics: a full-context prefill timed per ubatch chunk (the
   chunk series is the marginal cost curve — its slope is the attention
   term; its cumulative sum is TTFT vs depth), then decode rate at the
   reached depth and at fills below it, walked down the already-primed cache
   by trimming. The spawn also reports the model **geometry**: scalars,
   per-layer attention typing, tensor inventory, and the allocator's actual
   buffer sizes — including a **memory cost curve**, the breakdown
   re-measured at a ladder of context sizes. Counted by the runtime, never
   hand-maintained.
3. **Job** — one real end-to-end task, the per-machine check that the
   sweep-derived parameters reproduce an actual workload. The only spawn the
   memory sampler watches.

One process = one measurement unit — it loads at most one model, on one
provider. Memory stays attributable (one clean timeline per address space,
sampled from outside), failures isolate, and the exe stays simple: no provider
loop, no model cache; the harness owns the matrix.

**Nothing is measured twice.** A prompt is ingested ubatch-by-ubatch anyway,
so separate prefill points at doubling lengths would re-run the same loop —
the sweep instead times the chunks of one pass, and the decode ladder reuses
that pass's primed cache. Under the soft sweep budget (default 90 s) the
ladder stops between items: on slow silicon the measured envelope shrinks
instead of the time growing, and everything measured is emitted. Probe
points repeat adaptively (spread ≤ 5% judged from 3 repeats, capped at 5; a
≥ 20 s point runs once); the job keeps fixed **K in-process iterations × S
spawns** (defaults 2×1).

```
models.yaml + models/<M>/gguf/ ──► harness
                                   ├─ <exe> providers --model …      → [cpu, cuda, …]
   per provider:                   ├─ spawn <exe> probe --ep <p>     → ceilings (GEMM, copies)
   per cell (gate → sweep → job):  ├─ spawn <exe> run   … (gate)     → healthy?
                                   ├─ spawn <exe> sweep …            → geometry + curve points
   sample rss/vram every ~10 ms ───┤─ spawn <exe> run   … --iters K  → job events   (×S spawns)
   align samples to event windows ─┘
   aggregate ──► results/          (sweep/probe points carry adaptive repeats)
```

## The contract

The harness and the backends couple through exactly three things
(detail: [CLAUDE.md](CLAUDE.md), field-level truth: [`schema/`](schema/)):

1. a **CLI** every exe exposes — `providers` / `run` / `sweep` / `probe` /
   `version`;
2. the **events schema** the exe emits on stdout (stderr is for logging;
   nothing is downloaded at runtime);
3. a **`backend.toml`** telling the harness how to invoke the exe.

Backends are otherwise free: own language, own build, own deps. A versioned
JSON schema — rather than a shared library interface — lets each side evolve
independently and fail loudly at the seam: the harness validates events on the
way in and results on the way out. `bench check` conformance-tests a built exe.

## Measurement model

**Self-timed compute, externally-observed memory.** The exe times its own ops
with a monotonic clock; the harness samples memory over the process tree and
never touches the measured process. Only the job spawns are sampled — the
sampled footprint is the validation tick against the allocator-reported
memory model; gate, sweep, and probe spawns run unsampled. A single
wall-clock anchor captured at startup maps events onto the memory timeline.

**Equal work, deterministically.** Greedy/argmax decode, exactly `nb_tokens`
per turn, EOS ignored — every config does the same token count. The exe pins
intra-op threads to physical cores. Thinking is disabled
(`enable_thinking=false` through the model's own jinja chat template), so a
reasoning model doesn't burn its decode budget inside `<think>…</think>`.

**One canonical loop.** The exe drives its library's
low-level primitives (not `generate()`), isolating prefill from the first
decode step — TTFT (prefill start → first token) is measured identically
everywhere; decode throughput is steady-state over steps 2..N.

**A real operating point.** The micro-batch is the deployment default
(`n_ubatch = 512`), never tied to context size — measured rates and compute
buffers describe how the stack actually ships. Every output records its
`n_ctx / n_batch / n_ubatch` in the geometry block, so each number is
qualified by the operating point that produced it.

**Comparison is within a quant label.** A label names stack-specific math
(`q4` = Q4_K_M); every events object embeds
the exact stack versions to qualify a result.

**Correctness gates timing; it is not a quality benchmark.** One trivial
three-turn brain-check runs once per `(model, provider)` before anything
timed — every turn must pass or the provider's timed cells are skipped.
`expect` strings on trivially-knowable prompts catch plumbing failures (wrong
template, misconfigured provider, degenerate decode). Accepted limitation: a
provider that passes the gate but degenerates at a long-context prefill isn't
caught.

**Slow cells aren't ground.** The exe honours a soft `--deadline-ms`; the
harness hard-kills a pathologically slow iteration and skips monotonically
costlier tasks once one is unusable. Unusable cells are listed explicitly —
split into too-slow vs errored — never given an invented number.

**Memory figures are phase footprints.** Prefill reports its high-water mark;
decode reports both a **peak** (what must fit on the device) and a
**sustained** median (what generation occupies steady-state) — they diverge
when a transient, e.g. an EP compile spike, rides into the decode window. The
reported footprint is RSS, with per-PID NVML VRAM as its own pool and
unified-memory GTT folded into RSS; the why and the platform traps live in
[harness/README.md](harness/README.md).

**Raw traces first, results second.** Inference is expensive; aggregation is
cheap and changes often. The harness persists raw per-spawn traces
(`<backend>-raw.json.gz`), then derives the `[p50, max]` results from them —
`bench aggregate` re-runs only that second step, so a metric change is a
re-aggregate, not a re-run.
