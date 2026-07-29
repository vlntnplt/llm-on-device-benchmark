# tasks

What the benchmark runs. The harness resolves each YAML task into a flat task
JSON (documents inlined, budget explicit) and hands it to the exe via `--task`.

- `tasks.yaml` — the validation job (exactly one timed task).
- `brain_check.yaml` — the provider-health gate.
- `corpora/*.txt` — documents inlined into the job.

## One shape for everything

A task is `{name, max_context_length, messages}`. `system`/`user` turns carry
`content`; an `assistant` turn carries no content — it's a turn to
**generate** `nb_tokens` (greedy, EOS ignored), optionally with an `expect`
list. A user turn's content may be `{document: corpora/x.txt}`; the harness
inlines the text so every backend gets identical bytes and tokenizes them
itself. A task may hold several assistant turns; each turn prefills the full
re-rendered conversation (previous completions included) from a cleared
cache — no across-turn KV reuse to manage, every turn's events stand alone.

The harness never trims (it can't tokenize); it inlines documents in full and
warns loudly if a cell's rendered length overruns `max_context_length`, to be
fixed by hand here.

## The validation job

The cost curves come from the backend's synthetic sweeps; the one timed task
here is the end-to-end check that those curves predict a real workload.
`summarize-large`: a ~1700-token document (approximate — each backend
tokenizes with its own tokenizer), `max_context_length` 2048, 256 tokens
generated. Exactly one timed task must be defined; the harness refuses more.

## Gating

- **`expect` on a task** is a plumbing check, not a quality check: the decoded
  text must contain one of its strings. It catches a wrong chat template, a
  misconfigured provider, a degenerate decode — not bad summaries. Open-ended
  timed tasks carry none.
- **Brain-check** (`brain_check.yaml`) is the health gate: one trivial
  three-turn task run once per `(model, provider)` before anything else — a
  single spawn, one model load. Every turn's expect must pass or the provider
  is marked unhealthy and its sweep and job are skipped.

Accepted limitation: the gate validates plumbing up front; quality is not
re-checked at long context. This is a timing harness gated by a correctness
smoke test, not a quality benchmark.
