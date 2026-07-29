# ryzen-9-9950x-rtx-5080 — benchmark submission

| | |
|---|---|
| Host | Ryzen 9 9950X (RTX 5080) |
| OS | linux |
| CPU | AMD Ryzen 9 9950X 16-Core Processor (16C/32T) |
| GPU | NVIDIA GeForce RTX 5080 |
| Memory | 60.4 GB |
| Sampling | job: 1 spawns; sweep/probe: adaptive per point |

Status legend: `ok` (measured) · `too_slow` (backstop killed / below the floor) · `errored` (crash/OOM, no sample) · `skipped` · `unhealthy` (brain-check failed).

## ggml  (8 runs)

| provider | device | gemm TFLOP/s | d2d GB/s | probe |
|---|---|---|---|---|
| vulkan | NVIDIA GeForce RTX 5080 | 131.88 | 816.48 | ok |
| cpu | AMD Ryzen 9 9950X 16-Core Processor | 2.97 | 34.91 | ok |

| model | quant | provider | sweep | job |
|---|---|---|---|---|
| Qwen3.5-4B | q4 | cpu | ok (18 pts) | ok |
| Qwen3.5-4B | q4 | vulkan | ok (18 pts) | ok |
| gemma4-E2B | q4 | cpu | ok (19 pts) | ok |
| gemma4-E2B | q4 | vulkan | ok (19 pts) | ok |
| gemma4-E2B-qat | q2 | cpu | ok (19 pts) | ok |
| gemma4-E2B-qat | q2 | vulkan | ok (19 pts) | ok |
| gemma4-E4B | q4 | cpu | ok (19 pts) | ok |
| gemma4-E4B | q4 | vulkan | ok (19 pts) | ok |
