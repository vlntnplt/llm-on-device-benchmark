# core-ultra-5-125u-intel-graphics — benchmark submission

| | |
|---|---|
| Host | Core Ultra 5 125U (Intel Graphics) |
| OS | linux |
| CPU | Intel(R) Core(TM) Ultra 5 125U (12C/14T) |
| GPU | — |
| Memory | 15.1 GB 1-channel @ 4800 MT/s rank 1 |
| Sampling | job: 1 spawns; sweep/probe: adaptive per point |

Status legend: `ok` (measured) · `too_slow` (backstop killed / below the floor) · `errored` (crash/OOM, no sample) · `skipped` · `unhealthy` (brain-check failed).

## ggml  (8 runs)

| provider | device | gemm TFLOP/s | d2d GB/s | probe |
|---|---|---|---|---|
| vulkan | Intel(R) Graphics (MTL) | 0.47 | 56.92 | ok |
| cpu | Intel(R) Core(TM) Ultra 5 125U | 0.32 | 54.69 | ok |

| model | quant | provider | sweep | job |
|---|---|---|---|---|
| Qwen3.5-4B | q4 | cpu | ok (9 pts) | ok |
| Qwen3.5-4B | q4 | vulkan | ok (17 pts) | ok |
| gemma4-E2B | q4 | cpu | ok (17 pts) | ok |
| gemma4-E2B | q4 | vulkan | ok (10 pts) | ok |
| gemma4-E2B-qat | q2 | cpu | ok (16 pts) | ok |
| gemma4-E2B-qat | q2 | vulkan | ok (10 pts) | ok |
| gemma4-E4B | q4 | cpu | ok (11 pts) | ok |
| gemma4-E4B | q4 | vulkan | ok (9 pts) | ok |
