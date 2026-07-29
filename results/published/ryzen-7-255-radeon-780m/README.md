# ryzen-7-255-radeon-780m — benchmark submission

| | |
|---|---|
| Host | Ryzen 7 255 (Radeon 780M) |
| OS | linux |
| CPU | AMD Ryzen 7 255 w/ Radeon 780M Graphics (8C/16T) |
| GPU | — |
| Memory | 13.4 GB 2-channel @ 4800 MT/s rank 1 |
| Sampling | job: 1 spawns; sweep/probe: adaptive per point |

Status legend: `ok` (measured) · `too_slow` (backstop killed / below the floor) · `errored` (crash/OOM, no sample) · `skipped` · `unhealthy` (brain-check failed).

## ggml  (8 runs)

| provider | device | gemm TFLOP/s | d2d GB/s | probe |
|---|---|---|---|---|
| vulkan | AMD Radeon Graphics (RADV PHOENIX) | 3.5 | 64.09 | ok |
| cpu | AMD Ryzen 7 255 w/ Radeon 780M Graphics | 0.91 | 49.03 | ok |

| model | quant | provider | sweep | job |
|---|---|---|---|---|
| Qwen3.5-4B | q4 | cpu | ok (17 pts) | ok |
| Qwen3.5-4B | q4 | vulkan | ok (18 pts) | ok |
| gemma4-E2B | q4 | cpu | ok (19 pts) | ok |
| gemma4-E2B | q4 | vulkan | ok (19 pts) | ok |
| gemma4-E2B-qat | q2 | cpu | ok (19 pts) | ok |
| gemma4-E2B-qat | q2 | vulkan | ok (19 pts) | ok |
| gemma4-E4B | q4 | cpu | ok (19 pts) | ok |
| gemma4-E4B | q4 | vulkan | ok (19 pts) | ok |
