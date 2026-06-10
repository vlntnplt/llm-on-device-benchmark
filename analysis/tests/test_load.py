import json
import math
from pathlib import Path

import pytest

from bench_analysis import load_results

FIXTURES = Path(__file__).parent / "fixtures"


def test_merges_machines_by_directory():
    df = load_results(FIXTURES)
    assert set(df["machine"]) == {"3090-box", "m1-max"}  # label = subdir name
    assert set(df["backend"]) == {"ggml", "tjs"}


def test_flat_file_labels_by_host_not_gpu(tmp_path):
    """A results file straight under the root (no machine subdir) is labelled by its
    machine `host`, not by slugging the GPU."""
    (tmp_path / "ggml-results.json").write_text(
        json.dumps(
            {
                "schema_version": "1",
                "backend": "ggml",
                "machine": {
                    "host": "leaf-desktop",
                    "os": "linux",
                    "cpu": "x",
                    "cpu_cores": 16,
                    "cpu_threads": 32,
                    "gpus": ["NVIDIA RTX 5090"],
                },
                "iters": 1,
                "spawns": 1,
                "runs": [],
            }
        )
    )
    assert load_results(tmp_path).empty  # no runs, but it loaded without error
    # with a run present the label resolves to the host slug, not "nvidia-rtx-5090"
    doc = json.loads((tmp_path / "ggml-results.json").read_text())
    doc["runs"] = [
        {
            "provider": "CUDA",
            "device": "cuda:0",
            "model": "m",
            "quant": "q4",
            "healthy": False,
            "unhealthy_reason": "x",
            "vram_method": "nvml",
            "tasks": [],
        }
    ]
    (tmp_path / "ggml-results.json").write_text(json.dumps(doc))
    assert set(load_results(tmp_path)["machine"]) == {"leaf-desktop"}


def test_status_covers_every_cell():
    counts = load_results(FIXTURES)["status"].value_counts().to_dict()
    assert counts["ok"] == 2  # one timed task per machine
    assert counts["too_slow"] == 1  # 3090-box summarize-large (timed_out_tasks)
    assert counts["errored"] == 1  # 3090-box summarize-medium (errored_tasks: crash, not slow)
    assert counts["unhealthy"] == 1  # m1-max qwen3-4B failed its brain-check


def test_stats_explode_into_p50_max():
    df = load_results(FIXTURES)
    row = df[(df.machine == "3090-box") & (df.status == "ok")].iloc[0]
    assert row["decode_tps_p50"] == 80 and row["decode_tps_max"] == 75
    assert row["decode_vram_peak_mb_p50"] == 1700  # decode high-water
    assert row["decode_vram_sustained_mb_p50"] == 1650  # …vs the steady plateau
    assert row["prefill_vram_peak_mb_p50"] == 1600  # prompt-ingestion high-water


def test_null_stats_become_nan_not_zero():
    df = load_results(FIXTURES)
    row = df[(df.machine == "m1-max") & (df.status == "ok")].iloc[0]
    assert math.isnan(row["decode_vram_peak_mb_p50"])  # unified -> null -> NaN
    assert math.isnan(row["prefill_tps_p50"])


def test_schema_version_mismatch_is_loud(tmp_path):
    (tmp_path / "ggml-results.json").write_text(
        json.dumps(
            {
                "schema_version": "99",
                "backend": "ggml",
                "machine": {"os": "linux", "cpu": "x", "gpus": []},
                "iters": 1,
                "spawns": 1,
                "runs": [],
            }
        )
    )
    with pytest.raises(ValueError, match="schema_version"):
        load_results(tmp_path)


def test_empty_dir_returns_empty_frame(tmp_path):
    assert load_results(tmp_path).empty
