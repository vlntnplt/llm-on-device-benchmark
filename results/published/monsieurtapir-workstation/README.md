# monsieurtapir-workstation — benchmark submission

| | |
|---|---|
| Host | monsieurtapir-workstation |
| OS | linux |
| CPU | AMD Ryzen 9 9950X 16-Core Processor (16C/32T) |
| GPU | NVIDIA GeForce RTX 5080 |
| Sampling | 2 iters × 2 spawns (timing n = 4 per cell) |

Status legend: `ok` (timed) · `too_slow` (timed out / below the floor) · `errored` (crash/OOM, no sample) · `unhealthy` (brain-check failed).

## ggml  (6 runs)

| model | quant | provider | summarize-small | summarize-medium | summarize-large |
|---|---|---|---|---|---|
| Qwen3.5-4B | q4 | cpu | ok | ok | ok |
| Qwen3.5-4B | q4 | vulkan | ok | ok | ok |
| gemma4-E2B | q4 | cpu | ok | ok | ok |
| gemma4-E2B | q4 | vulkan | ok | ok | ok |
| gemma4-E4B | q4 | cpu | ok | ok | ok |
| gemma4-E4B | q4 | vulkan | ok | ok | ok |

## tjs  (9 runs)

| model | quant | provider | summarize-small | summarize-medium | summarize-large |
|---|---|---|---|---|---|
| Qwen3.5-4B | q4 | cpu | ok | ok | ok |
| Qwen3.5-4B | q4 | cuda | ok | ok | ok |
| Qwen3.5-4B | q4 | webgpu | ok | ok | ok |
| gemma4-E2B | q4 | cpu | ok | ok | ok |
| gemma4-E2B | q4 | cuda | ok | ok | ok |
| gemma4-E2B | q4 | webgpu | ok | ok | too_slow |
| gemma4-E4B | q4 | cpu | ok | ok | ok |
| gemma4-E4B | q4 | cuda | ok | ok | ok |
| gemma4-E4B | q4 | webgpu | ok | ok | too_slow |
