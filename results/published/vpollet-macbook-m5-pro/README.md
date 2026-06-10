# vpollet-macbook-m5-pro — benchmark submission

| | |
|---|---|
| Host | Mac.ht.home |
| OS | macos |
| CPU | Apple M5 Pro (18C/18T) |
| GPU | Apple M5 Pro |
| Sampling | 2 iters × 2 spawns (timing n = 4 per cell) |

Status legend: `ok` (timed) · `too_slow` (timed out / below the floor) · `errored` (crash/OOM, no sample) · `unhealthy` (brain-check failed).

## ggml  (6 runs)

| model | quant | provider | summarize-small | summarize-medium | summarize-large |
|---|---|---|---|---|---|
| Qwen3.5-4B | q4 | cpu | ok | ok | ok |
| Qwen3.5-4B | q4 | mtl | ok | ok | ok |
| gemma4-E2B | q4 | cpu | ok | ok | ok |
| gemma4-E2B | q4 | mtl | ok | ok | ok |
| gemma4-E4B | q4 | cpu | ok | ok | ok |
| gemma4-E4B | q4 | mtl | ok | ok | ok |

## tjs  (9 runs)

| model | quant | provider | summarize-small | summarize-medium | summarize-large |
|---|---|---|---|---|---|
| Qwen3.5-4B | q4 | coreml | ok | ok | ok |
| Qwen3.5-4B | q4 | cpu | ok | ok | ok |
| Qwen3.5-4B | q4 | webgpu | ok | ok | ok |
| gemma4-E2B | q4 | coreml | ok | ok | ok |
| gemma4-E2B | q4 | cpu | ok | ok | ok |
| gemma4-E2B | q4 | webgpu | ok | ok | ok |
| gemma4-E4B | q4 | coreml | ok | ok | ok |
| gemma4-E4B | q4 | cpu | ok | ok | ok |
| gemma4-E4B | q4 | webgpu | ok | ok | ok |
