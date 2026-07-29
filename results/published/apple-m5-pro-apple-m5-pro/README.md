# apple-m5-pro-apple-m5-pro — benchmark submission

| | |
|---|---|
| Host | apple-m5-pro-apple-m5-pro |
| OS | macos |
| CPU | Apple M5 Pro (18C/18T) |
| GPU | Apple M5 Pro |
| Memory | 48 GB |
| Sampling | job: 1 spawns; sweep/probe: adaptive per point |

Status legend: `ok` (measured) · `too_slow` (backstop killed / below the floor) · `errored` (crash/OOM, no sample) · `skipped` · `unhealthy` (brain-check failed).

## ggml  (8 runs)

| provider | device | gemm TFLOP/s | d2d GB/s | probe |
|---|---|---|---|---|
| mtl:0 | Apple M5 Pro | 7.27 | 140.28 | ok |
| cpu:0 | Apple M5 Pro | 1.18 | 174.99 | ok |

| model | quant | provider | sweep | job |
|---|---|---|---|---|
| Qwen3.5-4B | q4 | cpu:0 | ok (18 pts) | ok |
| Qwen3.5-4B | q4 | mtl:0 | ok (18 pts) | ok |
| gemma4-E2B | q4 | cpu:0 | ok (19 pts) | ok |
| gemma4-E2B | q4 | mtl:0 | ok (19 pts) | ok |
| gemma4-E2B-qat | q2 | cpu:0 | ok (19 pts) | ok |
| gemma4-E2B-qat | q2 | mtl:0 | ok (19 pts) | ok |
| gemma4-E4B | q4 | cpu:0 | ok (17 pts) | ok |
| gemma4-E4B | q4 | mtl:0 | ok (19 pts) | ok |
