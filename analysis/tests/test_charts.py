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
    for b, p, ttft, tps in [("ggml", "cpu", 1000.0, 10.0), ("ggml", "vulkan", 100.0, 40.0)]
] + [
    dict(machine="box", backend="ggml", provider="cuda", model="A", quant="q4",
         task="summarize-small", status="too_slow",
         cpu="AMD Ryzen 9 9950X 16-Core Processor", gpu="NVIDIA GeForce RTX 5080"),
]


MEM_ROWS = [
    dict(model="A", config=c, n_ctx=n,
         weights_mb=600.0, kv_mb=40.0 * n / 512, compute_mb=50.0)
    for c in ("ggml-cpu", "ggml-vulkan") for n in (512, 2048, 8192)
]


@pytest.fixture
def prepared():
    return prep.prepare(pd.DataFrame(ROWS))


def test_every_builder_emits_a_spec(prepared):
    df, task_order = prepared
    ok = df[df.status == "ok"]
    sliced = ok[ok.task == "summarize-small"]

    bars, ticks = prep.memory_model(pd.DataFrame(MEM_ROWS), sliced)
    specs = [
        charts.stacked(prep.time_phases(sliced), charts.TIME_COLORS, "t",
                       dnf=prep.failures(df[df.task == "summarize-small"])),
        charts.stacked(bars, charts.MEMORY_COLORS, "t", ticks=ticks),
    ]
    for spec in specs:
        assert isinstance(spec.to_dict(), dict)


def test_stacked_tick_rows_carry_no_object_nan(prepared):
    """The tick/dnf rows join the bar rows in one dataset; NaN left in an
    object-dtype column (phase, label) would escape Altair's sanitization and
    break the HTML export."""
    df, _ = prepared
    ok = df[df.status == "ok"]
    sliced = ok[ok.task == "summarize-small"]
    bars, ticks = prep.memory_model(pd.DataFrame(MEM_ROWS), sliced)
    spec = charts.stacked(bars, charts.MEMORY_COLORS, "t", ticks=ticks,
                          dnf=prep.failures(df[df.task == "summarize-small"])).to_dict()
    rows = next(iter(spec["datasets"].values()))
    assert all(r.get("phase") == r.get("phase") for r in rows)  # no NaN
    assert any(r.get("is_tick") for r in rows)


def test_palettes_cover_the_phase_labels():
    assert list(charts.TIME_COLORS) == prep.TIME_PHASES
    assert list(charts.MEMORY_COLORS) == prep.MEMORY_PHASES


def test_coverage_styler_renders(prepared):
    pytest.importorskip("jinja2")
    df, _ = prepared
    html = charts.coverage(df).to_html()
    assert "background-color" in html


def test_log_ticks_anchor_to_1_2_5_per_decade():
    ticks = charts._log_ticks(pd.Series([1.2, 47.0]))
    assert ticks == [1, 2, 5, 10, 20, 50]


def test_log_ticks_ignore_nan_and_nonpositive():
    # A log axis has nothing to say about 0 or NaN; 3.0 is the only real value,
    # so the kept ticks are the 1-2-5 steps within a factor of two of it.
    assert charts._log_ticks(pd.Series([math.nan, 0.0, 3.0])) == [2, 5]
    assert charts._log_ticks(pd.Series([math.nan])) is None
