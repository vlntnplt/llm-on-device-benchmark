# intel-ultra-5-125u — benchmark submission

| | |
|---|---|
| Host | intel-ultra-5-125u |
| OS | linux |
| CPU | Intel(R) Core(TM) Ultra 5 125U (12C/14T) |
| GPU | — |
| Sampling | 5 iters × 3 spawns (timing n = 15 per cell) |

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
