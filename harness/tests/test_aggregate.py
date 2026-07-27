"""Producer/consumer aggregation.

Synthesize the things spawns produce — run/sweep/probe events objects and
wall-stamped sample series — and check that `aggregate.build` turns the raw
into schema-valid results, that memory buckets by the events' own phase spans,
that sweep/probe points reduce to the declared throughputs, and that going
through gzip+JSON re-aggregates byte-identically.
"""

from __future__ import annotations

import argparse
import gzip
import json

from bench import aggregate, schema
from bench.commands.aggregate import cmd_aggregate

# Anchor: wall = W0 + (mono - 0). Keeping mono small makes the wall stamps readable.
W0 = 1_000_000_000


def _wall(mono: int) -> int:
    return W0 + mono


def _geometry() -> dict:
    return {
        "n_layer": 2,
        "n_embd": 64,
        "n_head": 4,
        "n_head_kv": 2,
        "n_swa": 0,
        "shared_kv_layers": 0,
        "n_ctx_train": 4096,
        "n_params": 1_000_000,
        "file_bytes": 600_000,
        "context": {"n_ctx": 2048, "n_batch": 2048, "n_ubatch": 512},
        "layers": [{"kind": "full", "window": 0}, {"kind": "recurrent", "window": 0}],
        "tensors": {
            "embedding": {"params": 100_000, "bytes": 60_000},
            "body": {"params": 900_000, "bytes": 540_000},
            "head": {"params": 0, "bytes": 0},
            "tied_head": True,
        },
        "buffers": [
            {"name": "CUDA0", "model_bytes": 540_000, "context_bytes": 4096, "compute_bytes": 1024}
        ],
    }


def _base(mode: str) -> dict:
    return {
        "schema_version": "2",
        "backend": "ggml",
        "mode": mode,
        "provider": "cuda",
        "device": "cuda:0",
        "versions": {"threads": 8},
        "anchor": {"wall_unix_ns": W0, "mono_ns": 0},
    }


def _run_events(*, healthy: bool = True) -> dict:
    """One iteration: load spans, a prefill batch, a 4-token decode, a turn-end.
    Spans are in mono_ns; the sampler's wall stamps line up via the anchor."""
    return {
        **_base("run"),
        "model": "demo",
        "quant": "q4",
        "task": "t",
        "healthy": healthy,
        "load": [
            {"type": "model-load", "start_ns": 0, "end_ns": 100},
            {"type": "context-init", "start_ns": 200, "end_ns": 300},
            {"type": "warmup", "start_ns": 400, "end_ns": 500},
        ],
        "geometry": _geometry(),
        "iterations": [
            {
                "events": [
                    {
                        "type": "prefill",
                        "context_size": 0,
                        "tokens_count": 10,
                        "start_ns": 1000,
                        "end_ns": 2000,
                    },
                    {
                        "type": "decode",
                        "context_size": 10,
                        "tokens_count": 4,
                        "token_ns": [2100, 3000, 4000, 5000],
                        "start_ns": 2000,
                        "end_ns": 5000,
                    },
                    {
                        "type": "turn-end",
                        "completion": "ok",
                        "expect_pass": True,
                        "start_ns": 5000,
                        "end_ns": 5001,
                    },
                ]
            }
        ],
    }


def _sweep_events() -> dict:
    """Two chunks of the instrumented pass (512 tokens in 1 s at context 0,
    then 1.5 s at context 512 — the marginal cost rising) and one decode point
    (4 tokens over 3 s steady → 1 tok/s)."""
    s = 1_000_000_000  # 1 second in ns
    return {
        **_base("sweep"),
        "model": "demo",
        "quant": "q4",
        "load": [{"type": "model-load", "start_ns": 0, "end_ns": 100}],
        "geometry": _geometry(),
        "prefill_chunks": [
            {"type": "prefill", "context_size": 0, "tokens_count": 512,
             "start_ns": 0, "end_ns": s},
            {"type": "prefill", "context_size": 512, "tokens_count": 512,
             "start_ns": 2 * s, "end_ns": int(3.5 * s)},
        ],
        "decode_points": [
            {"kv_fill": 1024, "tokens": 4,
             "repeats": [{"token_ns": [10 * s, 11 * s, 12 * s, 13 * s],
                          "start_ns": 10 * s, "end_ns": 13 * s}]},
        ],
    }


def _probe_events() -> dict:
    """One GEMM point moving 2·64³ FLOPs in 1 s and one d2d copy of 1 GB in 1 s
    (→ 2 GB/s of read+write traffic)."""
    s = 1_000_000_000
    return {
        **_base("probe"),
        "gemm": [{"m": 64, "n": 64, "k": 64, "dtype": "f16",
                  "repeats": [{"start_ns": 0, "end_ns": s}]}],
        "copy": [{"kind": "d2d", "bytes": 10**9, "repeats": [{"start_ns": 0, "end_ns": s}]},
                 {"kind": "h2d", "bytes": 10**9, "repeats": [{"start_ns": 0, "end_ns": s}]}],
    }


def _samples() -> list[dict]:
    """Two load-time samples (outside every phase span — must not leak into the
    phase stats) + one prefill + three decode. The first decode tick carries an RSS
    spike still draining out of prefill, so decode peak (1200) and sustained (the
    1000 plateau) differ."""

    def s(mono, rss, vram):
        return {"t": _wall(mono), "rss": rss, "vram": vram}

    return [
        s(150, 800e6, 1500e6),  # during load [100,200] — outside any timed phase
        s(350, 900e6, 1500e6),  # during context-init [300,400] — ditto
        s(1500, 950e6, 1900e6),  # prefill  [1000,2000] — KV + activation spike
        s(2500, 1200e6, 1750e6),  # decode   [2000,5000] — prefill spike draining
        s(3500, 1000e6, 1750e6),  # …the plateau decode actually sits on
        s(4500, 1000e6, 1750e6),
    ]


def _trace(events: dict | None = None) -> dict:
    return {"events": _run_events() if events is None else events, "samples": _samples()}


def _raw(cells: list[dict], probes: list[dict] | None = None) -> dict:
    return {
        "schema_version": "2",
        "backend": "ggml",
        "machine": {
            "host": "test-box",
            "os": "linux",
            "cpu": "x",
            "cpu_cores": 2,
            "cpu_threads": 4,
            "gpus": ["NVIDIA RTX 3090"],
            "memory": {"total_gb": 32.0, "channels": 2, "configured_mts": 4800,
                       "rated_mts": 5600, "rank": 1,
                       "dimms": [{"size_gb": 16, "configured_mts": 4800,
                                  "rated_mts": 5600, "rank": 1}] * 2},
        },
        # The run box's sources, as `bench run` records them. Aggregation derives
        # vram_method from THIS, never from the host running the tests.
        "sampling": {"nvml": True},
        "job_spawns": 2,
        "job_iters": 5,
        "probes": probes if probes is not None else
        [{"provider": "cuda", "trace": {"events": _probe_events(), "samples": []}}],
        "cells": cells,
    }


def _healthy_cell() -> dict:
    return {
        "model": "demo",
        "quant": "q4",
        "provider": "cuda",
        "healthy": True,
        "reason": None,
        "cold_ms": 123.0,
        "gate_spawns": [_trace()],
        "sweep": {"status": "ok", "trace": {"events": _sweep_events(), "samples": []}},
        "job": {"task": "summarize-large", "status": "ok", "spawns": [_trace()]},
    }


def test_build_is_schema_valid():
    results = aggregate.build(_raw([_healthy_cell()]))
    schema.validate_results(results)  # raises if not valid
    assert results["schema_version"] == "2"
    assert results["machine"]["memory"]["configured_mts"] == 4800


def test_probe_reduces_to_declared_throughputs():
    """GEMM tflops = 2·m·n·k / dt; a d2d copy counts read+write traffic (×2),
    h2d counts the payload once."""
    probes = aggregate.build(_raw([_healthy_cell()]))["probes"]
    assert len(probes) == 1 and probes[0]["status"] == "ok"
    gemm = probes[0]["gemm"][0]
    assert gemm["tflops_p50"] == round(2 * 64**3 / 1e12, 2)
    kinds = {c["kind"]: c["gbs_p50"] for c in probes[0]["copy"]}
    assert kinds["d2d"] == 2.0 and kinds["h2d"] == 1.0


def test_failed_probe_is_errored_not_invented():
    raw = _raw([_healthy_cell()], probes=[{"provider": "cuda",
                                           "trace": {"events": None, "samples": []}}])
    probe = aggregate.build(raw)["probes"][0]
    assert probe["status"] == "errored" and probe["gemm"] == [] and probe["copy"] == []


def test_sweep_chunks_pass_through_and_decode_reduces():
    """Chunks carry their exact single-pass cost at their context; the decode
    point's steady state is 3 tokens over 3 s → 1 tok/s."""
    sweep = aggregate.build(_raw([_healthy_cell()]))["runs"][0]["sweep"]
    assert sweep["status"] == "ok"
    assert sweep["prefill"][0] == {"context": 0, "tokens": 512, "ms": 1000.0}
    assert sweep["prefill"][1] == {"context": 512, "tokens": 512, "ms": 1500.0}
    d0 = sweep["decode"][0]
    assert (d0["kv_fill"], d0["tps_p50"], d0["n_reps"]) == (1024, 1.0, 1)


def test_geometry_carried_from_sweep():
    run = aggregate.build(_raw([_healthy_cell()]))["runs"][0]
    assert run["geometry"]["n_layer"] == 2
    assert run["geometry"]["context"]["n_ubatch"] == 512


def test_memory_split_by_phase_peak_and_sustained():
    """Memory is bucketed by the events' own prefill/decode spans. The prefill VRAM
    spike (1900) lands in prefill_vram; the decode plateau (1750) in decode_vram —
    they don't bleed into each other. Within decode, the spike riding on the first
    tick (1200) is the peak, while sustained is the plateau median (1000) — the
    transient must not masquerade as the steady working set."""
    mem = aggregate.build(_raw([_healthy_cell()]))["runs"][0]["job"]["memory"]
    assert mem["prefill_rss_peak_mb"] == [950, 950]
    assert mem["prefill_vram_peak_mb"] == [1900, 1900]
    assert mem["decode_rss_peak_mb"] == [1200, 1200]
    assert mem["decode_rss_sustained_mb"] == [1000, 1000]
    assert mem["decode_vram_peak_mb"] == [1750, 1750]
    assert mem["decode_vram_sustained_mb"] == [1750, 1750]


def test_phase_memory_null_when_no_tick_lands():
    """A prefill with no sample in its span → null memory, never invented."""
    cell = _healthy_cell()
    for tr in cell["job"]["spawns"]:
        tr["samples"] = [s for s in tr["samples"] if not (W0 + 1000 <= s["t"] <= W0 + 2000)]
    mem = aggregate.build(_raw([cell]))["runs"][0]["job"]["memory"]
    assert mem["prefill_rss_peak_mb"] is None and mem["prefill_vram_peak_mb"] is None
    assert mem["decode_vram_peak_mb"] == [1750, 1750]  # decode untouched


def test_methods_come_from_raw_not_aggregating_host():
    """vram_method must survive re-aggregation on ANY box — it describes the run
    box. A Mac raw must stay `unified` even when this test runs on Linux; a
    no-NVML raw must stay `n/a` even on an NVIDIA box."""
    mac = _raw([_healthy_cell()])
    mac["machine"]["os"] = "macos"
    mac["sampling"] = {"nvml": False}
    assert aggregate.build(mac)["runs"][0]["vram_method"] == "unified"

    apu = _raw([_healthy_cell()])
    apu["sampling"] = {"nvml": False}
    # no NVML on the run box → vram never read
    assert aggregate.build(apu)["runs"][0]["vram_method"] == "n/a"


def test_cold_start_attributed_to_the_job():
    cell = _healthy_cell()
    metrics = aggregate.build(_raw([cell]))["runs"][0]["job"]["metrics"]
    assert metrics["cold_start_ms"] == [123.0, 123.0]


def test_unhealthy_cell_skips_sweep_and_job():
    cell = {
        "model": "demo",
        "quant": "q4",
        "provider": "cuda",
        "healthy": False,
        "reason": "brain-check decode mismatch",
        "cold_ms": None,
        "gate_spawns": [_trace(_run_events(healthy=False))],
        "sweep": {"status": "skipped", "trace": None},
        "job": {"task": "summarize-large", "status": "skipped", "spawns": []},
    }
    run = aggregate.build(_raw([cell]))["runs"][0]
    assert run["healthy"] is False
    assert run["unhealthy_reason"] == "brain-check decode mismatch"
    assert run["sweep"]["status"] == "skipped" and run["sweep"]["prefill"] == []
    assert run["job"]["status"] == "skipped" and "metrics" not in run["job"]
    schema.validate_results(aggregate.build(_raw([cell])))


def test_too_slow_job_keeps_status_without_metrics():
    """A job below the floor keeps its traces (they inform vram_method) but is
    not scored — status carries the verdict, nothing is invented."""
    cell = _healthy_cell()
    cell["job"]["status"] = "too_slow"
    run = aggregate.build(_raw([cell]))["runs"][0]
    assert run["job"]["status"] == "too_slow" and "metrics" not in run["job"]
    schema.validate_results(aggregate.build(_raw([cell])))


def test_partial_sweep_points_survive_a_failed_sweep():
    """A sweep that died mid-way keeps the points it completed — a partial sweep
    still informs the fit."""
    cell = _healthy_cell()
    cell["sweep"]["status"] = "errored"
    sweep = aggregate.build(_raw([cell]))["runs"][0]["sweep"]
    assert sweep["status"] == "errored" and len(sweep["prefill"]) == 2


def test_aggregate_cli_sets_and_backfills_host(tmp_path):
    """`bench aggregate` must produce schema-valid results even from a raw with
    no `machine.host`: --machine sets it, else the hostname backfills."""
    raw = _raw([_healthy_cell()])
    del raw["machine"]["host"]
    path = tmp_path / "ggml-raw.json.gz"
    with gzip.open(path, "wt", encoding="utf-8") as fh:
        json.dump(raw, fh)

    cmd_aggregate(argparse.Namespace(raw=[path], machine="rescued-box"))
    out = json.loads((tmp_path / "ggml-results.json").read_text())
    assert out["machine"]["host"] == "rescued-box"  # override wins

    cmd_aggregate(argparse.Namespace(raw=[path], machine=None))
    out = json.loads((tmp_path / "ggml-results.json").read_text())
    assert out["machine"]["host"]  # backfilled (hostname), non-empty


def test_reaggregate_through_gzip_is_identical(tmp_path):
    """The whole point: persist raw, re-aggregate from disk, get the same
    results byte-for-byte — no re-inference."""
    raw = _raw([_healthy_cell()])
    live = aggregate.build(raw)

    path = tmp_path / "ggml-raw.json.gz"
    with gzip.open(path, "wt", encoding="utf-8") as fh:
        json.dump(raw, fh)
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        reloaded = aggregate.build(json.load(fh))

    assert json.dumps(reloaded, sort_keys=True) == json.dumps(live, sort_keys=True)
