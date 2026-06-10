"""Smoke tests: every builder produces a valid Vega-Lite spec from prep's
frames. Skipped wholesale when the `notebook` dependency group isn't installed
(altair/jinja2 are opt-in; the loader and prep stay pandas-only)."""

import math

import pandas as pd
import pytest

pytest.importorskip("altair")

from bench_analysis import charts, prep  # noqa: E402

ROWS = [
    dict(machine="box", backend=b, provider=p, model="A", quant="q4",
         task=t, status="ok",
         cpu="AMD Ryzen 9 9950X 16-Core Processor", gpu="NVIDIA GeForce RTX 5080",
         model_load_ms_p50=100.0, context_init_ms_p50=20.0, warmup_ms_max=30.0,
         ttft_ms_p50=ttft, completion_ms_p50=5 * ttft, decode_tps_p50=tps,
         decode_rss_sustained_mb_p50=1000.0, decode_vram_sustained_mb_p50=math.nan,
         decode_rss_peak_mb_max=1200.0, decode_vram_peak_mb_max=math.nan,
         prefill_rss_peak_mb_max=1500.0, prefill_vram_peak_mb_max=math.nan)
    for t in ("summarize-small", "summarize-large")
    for b, p, ttft, tps in [("ggml", "cpu", 1000.0, 10.0), ("ggml", "vulkan", 100.0, 40.0),
                            ("tjs", "cpu", 2000.0, 5.0)]
] + [
    dict(machine="box", backend="tjs", provider="webgpu", model="A", quant="q4",
         task="summarize-small", status="too_slow",
         cpu="AMD Ryzen 9 9950X 16-Core Processor", gpu="NVIDIA GeForce RTX 5080"),
]


@pytest.fixture
def prepared():
    return prep.prepare(pd.DataFrame(ROWS))


def test_every_builder_emits_a_spec(prepared):
    df, task_order = prepared
    ok = df[df.status == "ok"]
    sliced = ok[ok.task == "summarize-small"]

    specs = [
        charts.stacked(prep.time_phases(sliced), charts.TIME_COLORS, "t",
                       dnf=prep.failures(df[df.task == "summarize-small"])),
        charts.stacked(prep.memory_phases(sliced), charts.MEMORY_COLORS, "t"),
        charts.status_bars(prep.status_cells(df)),
        charts.dumbbell(prep.best_of_backend(ok), task_order, "t", row="machine"),
        charts.dumbbell(prep.lane_time(ok), task_order, "t"),
        charts.dumbbell(prep.lane_memory(ok), task_order, "t",
                        value="peak_gb", value_title="peak (GB)"),
        charts.dumbbell(prep.fallback_cost(ok), task_order, "t",
                        row="machine", value="seconds", y="leg", hue="side",
                        colors=charts.CPU_GPU_COLORS, ref_x=1.0),
    ]
    for spec in specs:
        assert isinstance(spec.to_dict(), dict)


def test_palettes_cover_the_phase_labels():
    assert list(charts.TIME_COLORS) == prep.TIME_PHASES
    assert list(charts.MEMORY_COLORS) == prep.MEMORY_PHASES


def test_coverage_styler_renders(prepared):
    pytest.importorskip("jinja2")
    df, _ = prepared
    html = charts.coverage(df).to_html()
    assert "background-color" in html
