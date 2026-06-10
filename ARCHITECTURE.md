# Architecture

A harness to compare on-device inference stacks on the same models and prompts,
over four axes: **load time, time-to-first-token, decode throughput, and memory**.

| backend | stack | language | artifact |
|---------|-------|----------|----------|
| `ggml`  | [llama.cpp](https://github.com/ggml-org/llama.cpp) | C++ (native) | one `.gguf`, runs on any provider |
| `tjs`   | [Transformers.js](https://github.com/huggingface/transformers.js) | JS (Node, `onnxruntime-node`) | ONNX dir, one file per dtype |

## Components

```
schema/            the wire contract — events (backend → harness) and results
                   (harness → analysis) JSON Schemas
tasks/             what the benchmark runs — timed catalog, corpora, the
                   brain-check provider-health gate
models.yaml        what it runs on — model registry + Hub fetch spec
backends/          one independently-buildable exe per stack (ggml, tjs)
harness/            backend-agnostic Python tool — enumerates cells, spawns
                   backends, samples memory, aggregates, publishes
analysis/          separate Python project — loads results into pandas,
                   compares across machines (marimo notebook)
results/published/ version-controlled benchmark submissions
```

Each component has a README with its specifics; this file is the map.

## One process = one cell

A **cell** is one `(model, variant, provider, task)`. The harness spawns exactly
one backend process per cell — that process loads the model once, on one
provider, and runs one task. Consequences:

- **Memory is attributable** — one load in one address space gives a single
  clean memory timeline, sampled from the outside.
- **Failures isolate** — a crash or unsupported provider takes down its own
  cell, nothing else.
- **The exe stays simple** — no provider loop, no model cache; the harness owns
  the matrix.

Repetition splits along two axes: **K in-process iterations** (default 5 —
cheap timing samples that pay model load and kernel/shader compilation once)
and **S process spawns** (default 3 — for load, cold-start, and memory
robustness). A timed cell costs S loads and yields S×K timing samples.

```
models.yaml + models/<M>/{gguf,onnx}/ ──► harness
                                          ├─ <exe> providers --model …   → [cpu, cuda, …]
   per cell, S spawns × K iters:          ├─ spawn <exe> run --ep <p> --task <t> --iters K ─► backend exe
   sample rss/vram every ~10 ms ──────────┤      ◄──────── one events object ────────         load+warmup ONCE,
   align samples to event windows ────────┘                                                   then K× prefill→decode
   aggregate ──► results/  ([p50, max])
```

## The contract

The harness and the backends couple through exactly three things
(detail: [CLAUDE.md](CLAUDE.md), field-level truth: [`schema/`](schema/)):

1. a **CLI** every exe exposes — `providers` / `run` / `version`;
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
never touches the measured process. A single wall-clock anchor captured at
startup maps events onto the memory timeline.

**Equal work, deterministically.** Greedy/argmax decode, exactly `nb_tokens`
per turn, EOS ignored — every backend does the same token count. Both exes pin
intra-op threads to physical cores. Thinking is disabled uniformly
(`enable_thinking=false` through the model's own jinja chat template), so a
reasoning model doesn't burn its decode budget inside `<think>…</think>`; the
rendered token ids match across stacks, and a cross-backend token-id check
verifies that.

**One canonical loop, two implementations.** Each exe drives its library's
low-level primitives (not `generate()`), isolating prefill from the first
decode step — TTFT (prefill start → first token) is measured identically;
decode throughput is steady-state over steps 2..N.

**Comparison is within a quant label.** A label is not identical math across
stacks (`ggml q4` = Q4_K_M, `tjs q4` = MatMulNBits); every events object embeds
the exact stack versions to qualify a result. GPU numbers compare *stacks on a
device*, never a shared kernel.

**Correctness gates timing; it is not a quality benchmark.** Three trivial
brain-check tasks run once per `(model, provider)` before its timed tasks —
all must pass or the provider's timed cells are skipped. `expect` strings on
trivially-knowable prompts catch plumbing failures (wrong template,
misconfigured provider, degenerate decode). Accepted limitation: a provider
that passes the gate but degenerates at a long-context prefill isn't caught.

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
