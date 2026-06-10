import math

import pandas as pd

from bench_analysis import prep

# One healthy row with every column prep touches; tests override what they probe.
BASE = dict(
    machine="box", backend="ggml", provider="cpu", model="A", quant="q4",
    task="summarize-small", status="ok",
    cpu="AMD Ryzen 9 9950X 16-Core Processor", gpu="NVIDIA GeForce RTX 5080",
    device="AMD Ryzen 9 9950X 16-Core Processor",
    model_load_ms_p50=100.0, context_init_ms_p50=20.0, warmup_ms_max=30.0,
    ttft_ms_p50=50.0, completion_ms_p50=250.0, decode_tps_p50=20.0,
    decode_rss_sustained_mb_p50=1000.0, decode_vram_sustained_mb_p50=math.nan,
    decode_rss_peak_mb_max=1200.0, decode_vram_peak_mb_max=math.nan,
    prefill_rss_peak_mb_max=1500.0, prefill_vram_peak_mb_max=math.nan,
)


def frame(*rows):
    return pd.DataFrame([{**BASE, **r} for r in rows])


def test_prepare_orders_tasks_ladder_first_then_unknowns():
    df, order = prep.prepare(frame(
        {"task": "summarize-large"}, {"task": "summarize-small"}, {"task": "zz-extra"},
    ))
    assert order == ["summarize-small", "summarize-large", "zz-extra"]
    assert list(df.task.cat.categories) == order and df.task.cat.ordered


def test_prepare_config_label_stays_short_for_one_machine_one_quant():
    df, _ = prep.prepare(frame({}), describe_machines=False)
    assert set(df.config) == {"ggml-cpu"}


def test_prepare_config_label_disambiguates_machine_and_quant():
    df, _ = prep.prepare(
        frame({"machine": "a"}, {"machine": "b", "quant": "q8"}),
        describe_machines=False,
    )
    assert set(df.config) == {"a · ggml-cpu q4", "b · ggml-cpu q8"}


def test_prepare_relabels_machines_and_keeps_submission():
    df, _ = prep.prepare(frame({}))
    assert set(df.machine) == {"Ryzen 9 9950X + RTX 5080"}
    assert set(df.submission) == {"box"}


def test_machine_labels_describe_hardware():
    labels = prep.machine_labels(frame(
        {"machine": "monsieurtapir-laptop",
         "cpu": "AMD Ryzen 5 PRO 230 w/ Radeon 760M Graphics",
         "gpu": "AMD Radeon 760M Graphics (RADV PHOENIX)"},
        {"machine": "vpollet-macbook-m5-pro", "cpu": "Apple M5 Pro", "gpu": "Apple M5 Pro"},
        {"machine": "cpu-only-box", "cpu": "Intel Core i5-1234", "gpu": "cpu"},
    ))
    assert labels["monsieurtapir-laptop"] == "Ryzen 5 230 + Radeon 760M"
    assert labels["vpollet-macbook-m5-pro"] == "Apple M5 Pro"  # unified: just the chip
    assert labels["cpu-only-box"] == "Core i5-1234"


def test_machine_labels_fall_back_to_accelerated_device():
    # Some machine blocks list no GPUs even though an iGPU ran (gpu = "cpu");
    # the vulkan provider's device string identifies it. The webgpu wrapper
    # ("webgpu (…cpu…)") must not win — it shortens to a digit-less name.
    _cpu = "AMD Ryzen 5 PRO 230 w/ Radeon 760M Graphics"
    labels = prep.machine_labels(frame(
        {"machine": "lap", "cpu": _cpu, "gpu": "cpu", "provider": "cpu", "device": _cpu},
        {"machine": "lap", "cpu": _cpu, "gpu": "cpu", "provider": "webgpu",
         "device": f"webgpu ({_cpu})"},
        {"machine": "lap", "cpu": _cpu, "gpu": "cpu", "provider": "vulkan",
         "device": "AMD Radeon 760M Graphics (RADV PHOENIX)"},
    ))
    assert labels["lap"] == "Ryzen 5 230 + Radeon 760M"


def test_machine_labels_dedupes_identical_hardware():
    labels = prep.machine_labels(frame({"machine": "a"}, {"machine": "b"}))
    assert labels == {"a": "Ryzen 9 9950X + RTX 5080 (a)",
                      "b": "Ryzen 9 9950X + RTX 5080 (b)"}


def test_time_phases_decode_is_completion_minus_ttft():
    df, _ = prep.prepare(frame({}), describe_machines=False)
    long = prep.time_phases(df)
    by_phase = long.set_index("phase").value
    assert by_phase["prefill"] == 50.0
    assert by_phase["decode"] == 200.0  # 250 completion − 50 TTFT
    assert set(long.phase) == set(prep.TIME_PHASES)


def test_memory_phases_drop_nan_vram_and_top_with_transient_peak():
    df, _ = prep.prepare(frame({}), describe_machines=False)
    long = prep.memory_phases(df)
    by_phase = long.set_index("phase").value
    assert "VRAM" not in by_phase  # NaN segment absent, not zero
    assert by_phase["RAM"] == 1000.0
    assert by_phase["transient peak"] == 200.0  # 1200 high-water − 1000 sustained


def test_memory_phases_clip_peak_below_sustained_to_zero():
    df, _ = prep.prepare(frame({"decode_rss_peak_mb_max": 900.0}),
                         describe_machines=False)
    by_phase = prep.memory_phases(df).set_index("phase").value
    assert by_phase["transient peak"] == 0.0


def test_shared_config_order_averages_ranks_and_puts_absentees_last():
    def phase_frame(*cfg_vals):
        return pd.DataFrame([{"model": "A", "config": c, "phase": "prefill", "value": v}
                             for c, v in cfg_vals])

    order = prep.shared_config_order([
        phase_frame(("fast", 1.0), ("mid", 2.0), ("flaky", 3.0)),
        phase_frame(("fast", 1.0), ("mid", 2.0)),  # flaky has no sample → last
    ])
    assert order == ["fast", "mid", "flaky"]


def test_failures_keeps_only_failed_configs_with_pretty_labels():
    df, _ = prep.prepare(frame(
        {}, {"provider": "webgpu", "status": "too_slow"},
        {"provider": "cuda", "status": "errored"},
    ), describe_machines=False)
    out = prep.failures(df)
    assert set(out.label) == {"too slow", "errored"}
    assert "ggml-cpu" not in set(out.config)  # the ok config isn't a failure


def test_status_cells_counts_unhealthy_once_per_config():
    df, _ = prep.prepare(frame(
        {}, {"backend": "tjs", "provider": "webgpu", "status": "unhealthy", "task": None},
    ), describe_machines=False)
    cells = prep.status_cells(df)
    row = cells[cells.status == "unhealthy"].iloc[0]
    assert row.who == "tjs · webgpu" and row.n == 1


def test_prepare_assigns_silicon_named_lanes():
    df, _ = prep.prepare(frame({}, {"provider": "vulkan"}))
    assert set(df.lane) == {"Ryzen 9 9950X · cpu", "RTX 5080 · gpu"}


def test_lane_labels_borrow_cpu_name_when_gpu_is_unified():
    lanes = prep.lane_labels(frame(
        {"machine": "mac", "cpu": "Apple M5 Pro", "gpu": "Apple M5 Pro"}))
    assert lanes["mac"] == {"cpu": "Apple M5 Pro · cpu", "gpu": "Apple M5 Pro · gpu"}


def test_lane_labels_dedupe_collisions_across_machines():
    lanes = prep.lane_labels(frame({"machine": "a"}, {"machine": "b"}))
    assert lanes["a"]["cpu"] == "Ryzen 9 9950X · cpu (a)"
    assert lanes["b"]["gpu"] == "RTX 5080 · gpu (b)"


def test_best_of_backend_picks_fastest_provider_and_pairs_backends():
    df, _ = prep.prepare(frame(
        # ggml: vulkan total 0.4 s beats cpu total 1.0 s
        {"provider": "vulkan"},
        {"provider": "cpu", "completion_ms_p50": 850.0},
        # tjs: one provider, total 0.8 s
        {"backend": "tjs", "completion_ms_p50": 650.0},
    ), describe_machines=False)
    best = prep.best_of_backend(df)
    assert len(best) == 2  # one winner per backend
    ggml = best[best.backend == "ggml"].iloc[0]
    assert ggml.provider == "vulkan" and ggml.total_s == 0.4
    assert ggml.n_backends == 2
    assert ggml.ratio == 2.0  # tjs 0.8 / ggml 0.4


def test_lane_time_matches_backends_within_a_lane_only():
    df, _ = prep.prepare(frame(
        # cpu lane: ggml 0.4 s vs tjs 0.8 s — a real matchup.
        {},
        {"backend": "tjs", "completion_ms_p50": 650.0},
        # gpu lane: ggml alone — a walkover, not a tie.
        {"provider": "vulkan"},
    ), describe_machines=False)
    best = prep.lane_time(df)
    cpu_lane = best[best.lane == "Ryzen 9 9950X · cpu"]
    assert set(cpu_lane.backend) == {"ggml", "tjs"}
    assert cpu_lane.ratio.iloc[0] == 2.0  # tjs 0.8 / ggml 0.4
    gpu_lane = best[best.lane == "RTX 5080 · gpu"].iloc[0]
    assert gpu_lane.n_backends == 1


def test_lane_memory_peaks_at_the_heavier_of_prefill_and_decode():
    df, _ = prep.prepare(frame(
        {},                                                          # prefill 1500 wins
        {"backend": "tjs", "decode_rss_peak_mb_max": 2000.0,         # decode wins
         "decode_vram_peak_mb_max": 48.0},
    ), describe_machines=False)
    mem = prep.lane_memory(df).set_index("backend")
    assert mem.loc["ggml", "peak_gb"] == 1500.0 / 1024
    assert mem.loc["tjs", "peak_gb"] == 2048.0 / 1024
    assert mem.loc["tjs", "ratio"] == mem.loc["tjs", "hi"] / mem.loc["tjs", "lo"]


def test_gpu_vs_cpu_pairs_cpu_with_fastest_accelerated():
    df, _ = prep.prepare(frame(
        {"ttft_ms_p50": 1000.0, "decode_tps_p50": 10.0},                      # cpu
        {"provider": "vulkan", "ttft_ms_p50": 100.0, "decode_tps_p50": 40.0,  # winner
         "completion_ms_p50": 50.0},
        {"provider": "opencl", "ttft_ms_p50": 200.0, "decode_tps_p50": 20.0},
        # a cpu-only machine has nothing to pair with → drops out
        {"machine": "cpu-only"},
    ), describe_machines=False)
    gvc = prep.gpu_vs_cpu(df)
    assert len(gvc) == 1
    row = gvc.iloc[0]
    assert row.provider_gpu == "vulkan"
    assert row.prefill_x == 10.0 and row.decode_x == 4.0
    assert row.completion_x == 5.0  # 250 ms cpu turn / 50 ms gpu turn
    assert row.ttft_s_cpu == 1.0 and row.ttft_s_gpu == 0.1


def test_fallback_cost_melts_pairs_into_dumbbell_legs():
    df, _ = prep.prepare(frame(
        {"ttft_ms_p50": 1000.0, "completion_ms_p50": 3000.0},               # cpu
        {"provider": "vulkan", "ttft_ms_p50": 100.0, "completion_ms_p50": 600.0},
    ), describe_machines=False)
    fc = prep.fallback_cost(df)
    assert len(fc) == 4  # 2 sides × 2 phases
    ttft = fc[fc.phase == "TTFT"].set_index("side")
    assert ttft.loc["cpu", "seconds"] == 1.0 and ttft.loc["gpu", "seconds"] == 0.1
    assert ttft.ratio.iloc[0] == 10.0  # the TTFT leg's gap label = prefill_x
    dec = fc[fc.phase == "decode"].set_index("side")
    assert dec.loc["cpu", "seconds"] == 2.0 and dec.loc["gpu", "seconds"] == 0.5
    assert set(fc.leg) == {"A · TTFT", "A · decode"}


def test_gpu_vs_cpu_ignores_other_backends():
    df, _ = prep.prepare(frame(
        {"ttft_ms_p50": 1000.0},
        {"backend": "tjs", "provider": "webgpu", "ttft_ms_p50": 1.0},
    ), describe_machines=False)
    assert prep.gpu_vs_cpu(df, backend="ggml").empty
