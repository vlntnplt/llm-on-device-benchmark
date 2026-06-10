# tasks

What the benchmark runs. The harness resolves each YAML task into a flat task
JSON (documents inlined, budget explicit) and hands it to the exe via `--task`.

- `tasks.yaml` — the timed catalog.
- `brain_check.yaml` — the provider-health gate.
- `corpora/*.txt` — documents inlined into the summarize tasks.

## One shape for everything

A task is `{name, max_context_length, messages}`. `system`/`user` turns carry
`content`; the single `assistant` turn carries no content — it's a turn to
**generate** `nb_tokens` (greedy, EOS ignored), optionally with an `expect`
list. A user turn's content may be `{document: corpora/x.txt}`; the harness
inlines the text so every backend gets identical bytes and tokenizes them
itself. Every task is single-turn: one prefill of the rendered prompt, one
decode — no across-turn KV reuse to manage.

The harness never trims (it can't tokenize); it inlines documents in full and
warns loudly if a cell's rendered length overruns `max_context_length`, to be
fixed by hand here.

## The timed catalog

`summarize-{small,medium,large}` sweeps prefill cost across a 4×-growing
prompt at a modest decode budget:

| task | `max_context_length` | prefill (≈tokens) | `nb_tokens` |
|------|---------------------|-------------------|-------------|
| small  | 512  | ~400  | 64  |
| medium | 1024 | ~850  | 128 |
| large  | 2048 | ~1700 | 256 |

Prefill counts are approximate — each backend tokenizes with its own tokenizer.

## Gating

- **`expect` on a task** is a plumbing check, not a quality check: the decoded
  text must contain one of its strings. It catches a wrong chat template, a
  misconfigured provider, a degenerate decode — not bad summaries. Open-ended
  timed tasks carry none.
- **Brain-check** (`brain_check.yaml`) is the health gate: three trivial tasks
  run once per `(model, provider)` before its timed tasks. All three must pass
  or the provider is marked unhealthy and its timed cells are skipped.

Accepted limitation: the gate validates plumbing up front; quality is not
re-checked on long-context timed tasks. This is a timing harness gated by a
correctness smoke test, not a quality benchmark.
