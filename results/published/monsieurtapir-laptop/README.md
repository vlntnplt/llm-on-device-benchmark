# monsieurtapir-laptop — benchmark submission

| | |
|---|---|
| Host | monsieurtapir-laptop |
| OS | linux |
| CPU | AMD Ryzen 5 PRO 230 w/ Radeon 760M Graphics (6C/12T) |
| GPU | — |
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

## tjs  (6 runs)

| model | quant | provider | summarize-small | summarize-medium | summarize-large |
|---|---|---|---|---|---|
| Qwen3.5-4B | q4 | cpu | ok | ok | ok |
| Qwen3.5-4B | q4 | webgpu | too_slow | too_slow | too_slow |
| gemma4-E2B | q4 | cpu | ok | ok | ok |
| gemma4-E2B | q4 | webgpu | too_slow | too_slow | too_slow |
| gemma4-E4B | q4 | cpu | ok | ok | ok |
| gemma4-E4B | q4 | webgpu | too_slow | too_slow | too_slow |

# Notes
- Had to run with environment variable `GGML_VK_DISABLE_FUSION=1` to work around a hang for Gemma E2B and E4B
- Laptop was overall unusable during the benchmark: it ran hot and apps froze a lot