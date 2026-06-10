"""Producer/consumer aggregation.

Synthesize the two things a spawn produces — an events object and a wall-stamped
sample series — and check that `aggregate.build` turns the raw cell into
schema-valid results, that memory buckets by the events' own phase spans, and
that going through gzip+JSON re-aggregates byte-identically.
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


def _events(*, healthy: bool = True) -> dict:
    """One iteration: load spans, a prefill batch, a 4-token decode, a turn-end.
    Spans are in mono_ns; the sampler's wall stamps line up via the anchor."""
    return {
        "schema_version": "1",
        "backend": "ggml",
        "provider": "cuda",
        "device": "cuda:0",
        "model": "demo",
        "quant": "q4",
        "task": "t",
        "versions": {"threads": 8},
        "anchor": {"wall_unix_ns": W0, "mono_ns": 0},
        "healthy": healthy,
        "load": [
            {"type": "model-load", "start_ns": 0, "end_ns": 100},
            {"type": "context-init", "start_ns": 200, "end_ns": 300},
            {"type": "warmup", "start_ns": 400, "end_ns": 500},
        ],
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


def _trace(healthy: bool = True) -> dict:
    return {"events": _events(healthy=healthy), "samples": _samples()}


def _raw(cells: list[dict]) -> dict:
    return {
        "schema_version": "1",
        "backend": "ggml",
        "machine": {
            "host": "test-box",
            "os": "linux",
            "cpu": "x",
            "cpu_cores": 2,
            "cpu_threads": 4,
            "gpus": ["NVIDIA RTX 3090"],
        },
        # The run box's sources, as `bench run` records them. Aggregation derives
        # vram_method from THIS, never from the host running the tests.
        "sampling": {"nvml": True},
        "iters": 1,
        "spawns": 1,
        "cells": cells,
    }


def _healthy_cell() -> dict:
    return {
        "model": "demo",
        "quant": "q4",
        "provider": "cuda",
        "healthy": True,
        "reason": None,
        "timed_out_tasks": [],
        "cold_ms": 123.0,
        "gate_spawns": [_trace()],
        "tasks": [{"task": "summarize-small", "spawns": [_trace()]}],
    }


def test_build_is_schema_valid():
    results = aggregate.build(_raw([_healthy_cell()]))
    schema.validate_results(results)  # raises if not valid
    assert results["schema_version"] == "1"
    assert results["machine"]["gpus"] == ["NVIDIA RTX 3090"]


def test_memory_split_by_phase_peak_and_sustained():
    """Memory is bucketed by the events' own prefill/decode spans. The prefill VRAM
    spike (1900) lands in prefill_vram; the decode plateau (1750) in decode_vram —
    they don't bleed into each other. Within decode, the spike riding on the first
    tick (1200) is the peak, while sustained is the plateau median (1000) — the
    transient must not masquerade as the steady working set."""
    mem = aggregate.build(_raw([_healthy_cell()]))["runs"][0]["tasks"][0]["memory"]
    assert mem["prefill_rss_peak_mb"] == [950, 950]
    assert mem["prefill_vram_peak_mb"] == [1900, 1900]
    assert mem["decode_rss_peak_mb"] == [1200, 1200]
    assert mem["decode_rss_sustained_mb"] == [1000, 1000]
    assert mem["decode_vram_peak_mb"] == [1750, 1750]
    assert mem["decode_vram_sustained_mb"] == [1750, 1750]


def test_phase_memory_null_when_no_tick_lands():
    """A prefill with no sample in its span → null memory, never invented."""
    cell = _healthy_cell()
    for tr in cell["tasks"][0]["spawns"]:
        tr["samples"] = [s for s in tr["samples"] if not (W0 + 1000 <= s["t"] <= W0 + 2000)]
    mem = aggregate.build(_raw([cell]))["runs"][0]["tasks"][0]["memory"]
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


def test_raw_without_sampling_block_infers_sources_from_data():
    """A raw missing the `sampling` block: VRAM samples > 0 can only have come
    from NVML, so the method reconstructs; all-zero VRAM means it was never
    read."""
    raw = _raw([_healthy_cell()])
    del raw["sampling"]
    assert aggregate.build(raw)["runs"][0]["vram_method"] == "nvml"

    apu = _raw([_healthy_cell()])
    del apu["sampling"]
    for tr in [*apu["cells"][0]["gate_spawns"], *apu["cells"][0]["tasks"][0]["spawns"]]:
        for s in tr["samples"]:
            s["vram"] = 0  # APU: weights in GTT (folded into rss), never in vram
    assert aggregate.build(apu)["runs"][0]["vram_method"] == "n/a"


def test_cold_start_attributed_once():
    cell = _healthy_cell()
    metrics = aggregate.build(_raw([cell]))["runs"][0]["tasks"][0]["metrics"]
    assert metrics["cold_start_ms"] == [123.0, 123.0]  # the cell's cold_ms, to its first task


def test_unhealthy_cell_has_no_tasks():
    cell = {
        "model": "demo",
        "quant": "q4",
        "provider": "cuda",
        "healthy": False,
        "reason": "brain-check decode mismatch",
        "timed_out_tasks": [],
        "cold_ms": None,
        "gate_spawns": [_trace(healthy=False)],
        "tasks": [],
    }
    run = aggregate.build(_raw([cell]))["runs"][0]
    assert run["healthy"] is False and run["tasks"] == []
    assert run["unhealthy_reason"] == "brain-check decode mismatch"


def test_too_slow_task_excluded_but_traces_count():
    """A task whose spawns all timed out stays in timed_out_tasks (not scored), but
    its traces still inform vram/gpu method detection via all_traces."""
    cell = _healthy_cell()
    cell["timed_out_tasks"] = ["summarize-large"]
    cell["tasks"].append({"task": "summarize-large", "spawns": [_trace()]})
    run = aggregate.build(_raw([cell]))["runs"][0]
    assert [t["task"] for t in run["tasks"]] == ["summarize-small"]
    assert run["timed_out_tasks"] == ["summarize-large"]


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
    """The whole point : persist raw, re-aggregate from disk, get the same
    results byte-for-byte — no re-inference."""
    raw = _raw([_healthy_cell()])
    live = aggregate.build(raw)

    path = tmp_path / "ggml-raw.json.gz"
    with gzip.open(path, "wt", encoding="utf-8") as fh:
        json.dump(raw, fh)
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        reloaded = aggregate.build(json.load(fh))

    assert json.dumps(reloaded, sort_keys=True) == json.dumps(live, sort_keys=True)
