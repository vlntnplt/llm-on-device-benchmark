# `tjs` backend — Transformers.js

The Node backend over [Transformers.js](https://github.com/huggingface/transformers.js)
(`@huggingface/transformers` on `onnxruntime-node`), driving a manual
`model.forward()` loop rather than `generate()`. Model artifact: the fetched
ONNX repo dir (`models/<name>/onnx/`), where `--quant` selects the dtype file
(`q4`→`onnx/model_q4.onnx`). Backend **key is `tjs`** (declared in
[`backend.toml`](backend.toml)); the directory is named for readability.
Contract: [ARCHITECTURE.md](../../ARCHITECTURE.md).

## Build

Prerequisites: Node ≥ 18 and `npm`.

```sh
cd backends/transformersjs
npm install            # @huggingface/transformers + onnxruntime-node (pinned)
npm run build          # esbuild bundles src/bench.js → dist/bench.js
node dist/bench.js version
```

`npm install` is the only step that touches the network; afterwards everything
runs offline (`env.allowRemoteModels = false`). esbuild leaves the native deps
(`onnxruntime-node`, `sharp`) external, so `node_modules/` must stay present at
run time. `dist/` is a build output (git-ignored) — rerun `npm run build`
after editing `src/`; the entry point must stay the one named in
`backend.toml` (**`dist/bench.js`**, invoked as `node {dir}/dist/bench.js`).

GPU EPs ride on whatever `onnxruntime-node` ships for your platform: `cuda`
(Linux x64 + CUDA 12 — `source cuda-env.sh` from the repo root puts the
CUDA-12 pip-wheel libs on the loader path) and the experimental `webgpu`. An
EP that can't load is caught by the harness's brain-check gate and its timed
cells are skipped.

Format with `npm run format` (prettier). Validate conformance from the repo
root: `uv run --project harness bench check --backend tjs --models models`.

## Contract notes specific to tjs

- `from_pretrained(dir, { dtype, device, session_options, local_files_only: true })`
  — always offline. `--quant` maps to `dtype` (`fp16`→`model_fp16.onnx`,
  `q8`→`model_quantized.onnx`, `q4`→`model_q4.onnx`); `--ep` maps to `device`.
  `--model` may be the repo dir or a resolved `onnx/model_*.onnx` inside it —
  the exe derives the dir holding the `tokenizer.json`/`config.json` markers.
- `providers` reports the devices `onnxruntime-node` supports on this
  platform, `cpu` first: `cpu`; `cuda` on Linux x64+CUDA12; `coreml` on macOS;
  experimental `webgpu`.
- The chat template is read from `tokenizer_config.json`'s `chat_template`
  (transformers.js does not load a side `chat_template.jinja`). Render via
  `apply_chat_template(msgs, { add_generation_prompt: true, enable_thinking: false })`
  — the template emits its own thinking-off block inline; nothing is
  hardcoded, so token ids match the other stacks.
- Manual forward loop: prefill all prompt tokens in one `model.forward`, then
  a `forward(1 tok)` + argmax greedy loop — isolating prefill from the first
  decode step so `prefill_tps` and `ttft_ms` are both reported.
- onnxruntime session intra-op threads pinned to physical cores
  (`session_options.intraOpNumThreads`); recorded in `versions`, alongside the
  `@huggingface/transformers` + `onnxruntime-node` + Node/V8 versions and
  platform/arch.
