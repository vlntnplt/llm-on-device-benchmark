# `ggml` backend — llama.cpp

The native backend over [llama.cpp](https://github.com/ggml-org/llama.cpp),
driving the low-level `llama.h` API (not `generate()`). Model artifact: a
single `.gguf` file that runs on any provider the build supports (`cpu`,
`cuda`, `metal`, …) — chosen at runtime via `--ep`. Contract:
[ARCHITECTURE.md](../../ARCHITECTURE.md).

## Build

Prerequisites: CMake ≥ 3.18, a C++17 compiler, `git`. A GPU build also needs
its toolchain (CUDA toolkit for `-DGGML_CUDA=ON`, Xcode for `-DGGML_METAL=ON`).

```sh
cd backends/ggml
cmake -B build -S .          # add -DGGML_CUDA=ON / -DGGML_VULKAN=ON / -DGGML_METAL=ON
cmake --build build -j
./build/bench-ggml version   # sanity check: prints versions JSON
```

`CMakeLists.txt` fetches its own dependencies at configure time, pinned for
reproducibility: llama.cpp at `LLAMACPP_GIT_TAG` (embedded as a subproject,
tests/tools/server off), CLI11 by SHA256; nlohmann/json comes from llama.cpp's
vendored copy. The first configure needs network (cached under `build/`
afterward); pass `-DLLAMACPP_SOURCE_DIR=<path>` (e.g. a
`third_party/llama.cpp` checkout) to build offline or hack on the stack. The
output is the exe named in [`backend.toml`](backend.toml) —
**`build/bench-ggml`** — keep the two in sync.

Format with `clang-format -i main.cpp`. Validate conformance from the repo
root: `uv run --project harness bench check --backend ggml --models models`.

## Contract notes specific to ggml

- **mmap stays off** (`use_mmap=false`) — the shipped deployment
  configuration. Weights are read into allocated buffers at load: the load
  phase pays the full file read, RSS carries the whole weight footprint, and
  the CPU backend's repacked quant tensors exist only in their repacked form
  (no mapped originals kept alongside). See
  [harness/README.md](../../harness/README.md) for how memory is reported.
- `providers` walks the ggml backend registry (`ggml_backend_reg_*`,
  `ggml_backend_dev_get_props`).
- Render prompts via `common_chat_templates_apply` (the model's own jinja
  template) with `enable_thinking=false`. Do **not** use
  `llama_chat_apply_template`: it's a non-jinja built-in approximation with no
  thinking knob that silently diverges from some real templates.
  `enable_thinking=false` lets the template emit its own thinking-off block
  inline; nothing is hardcoded, so the rendered token ids match the other
  stacks.
- Prefill is isolated from the first decode step, so `prefill_tps` and
  `ttft_ms` are both reported.
- `n_threads` / `n_threads_batch` pinned to physical cores; recorded in
  `versions`, alongside the llama.cpp commit, compiled backends + toolkit
  versions, build flags, and `use_mmap`.
- The binary is ad-hoc signed on macOS (`entitlements.plist`,
  `com.apple.security.get-task-allow`) so `task_for_pid` stays available to
  the sampler.
