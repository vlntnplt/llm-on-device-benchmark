"""The amend path: `merge_raw` folds one trace's cells into another only when
both measured the same experiment — same machine identity, stack versions, and
job shape — and otherwise refuses loudly. New cells append, re-measured cells
replace, probes stay the base's."""

from __future__ import annotations

import pytest

from bench.commands.merge import merge_raw


def _cell(model: str, quant: str = "q4", provider: str = "cpu", versions: dict | None = None):
    events = {"versions": versions or {"llama_cpp_commit": "abc123", "compiler": "GNU 16"}}
    return {
        "model": model,
        "quant": quant,
        "provider": provider,
        "healthy": True,
        "reason": None,
        "cold_ms": None,
        "gate_spawns": [{"events": events, "samples": []}],
        "sweep": {"status": "ok", "trace": {"events": events, "samples": []}},
        "job": {"task": "job", "status": "ok", "spawns": [{"events": events, "samples": []}]},
    }


def _raw(cells: list[dict], probes: list[dict] | None = None, **over):
    raw = {
        "schema_version": "2",
        "backend": "ggml",
        "machine": {
            "host": "box",
            "os": "linux",
            "cpu": "cpu0",
            "cpu_cores": 8,
            "cpu_threads": 16,
            "gpus": [],
            "memory": {"total_gb": 16.0},
        },
        "sampling": {"nvml": False, "drm": True},
        "job_spawns": 1,
        "job_iters": 2,
        "probes": probes if probes is not None else [],
        "cells": cells,
    }
    raw.update(over)
    return raw


def test_new_cells_append_and_duplicates_replace():
    old = _cell("m1")
    base = _raw([old, _cell("m1", provider="vulkan")])
    replacement = _cell("m1")
    replacement["cold_ms"] = 42.0
    merged = merge_raw(base, _raw([replacement, _cell("m2")]))
    keys = [(c["model"], c["provider"]) for c in merged["cells"]]
    assert keys == [("m1", "cpu"), ("m1", "vulkan"), ("m2", "cpu")]
    assert merged["cells"][0]["cold_ms"] == 42.0  # replaced, not kept


def test_base_probes_win_new_providers_land():
    base = _raw([_cell("m1")], probes=[{"provider": "cpu", "trace": {"events": None}}])
    new = _raw(
        [_cell("m2")],
        probes=[
            {"provider": "cpu", "trace": {"events": None, "tag": "fresh"}},
            {"provider": "vulkan", "trace": {"events": None}},
        ],
    )
    merged = merge_raw(base, new)
    by_ep = {p["provider"]: p for p in merged["probes"]}
    assert by_ep["cpu"]["trace"] == {"events": None}  # base's kept
    assert set(by_ep) == {"cpu", "vulkan"}


@pytest.mark.parametrize(
    "over",
    [
        {
            "machine": {
                "host": "other-box",
                "os": "linux",
                "cpu": "cpu0",
                "cpu_cores": 8,
                "cpu_threads": 16,
                "gpus": [],
                "memory": {"total_gb": 16.0},
            }
        },
        {"job_spawns": 3},
        {"job_iters": 5},
        {"sampling": {"nvml": True, "drm": True}},
        {"backend": "other"},
        {"schema_version": "1"},
    ],
)
def test_different_experiment_refused(over):
    with pytest.raises(SystemExit):
        merge_raw(_raw([_cell("m1")]), _raw([_cell("m2")], **over))


def test_version_drift_refused():
    drifted = _cell("m2", versions={"llama_cpp_commit": "def456", "compiler": "GNU 16"})
    with pytest.raises(SystemExit):
        merge_raw(_raw([_cell("m1")]), _raw([drifted]))


def test_memory_config_difference_tolerated():
    new = _raw([_cell("m2")])
    new["machine"] = dict(new["machine"], memory={"total_gb": 16.0, "channels": 2})
    merged = merge_raw(_raw([_cell("m1")]), new)
    assert merged["machine"]["memory"] == {"total_gb": 16.0}  # base's kept
