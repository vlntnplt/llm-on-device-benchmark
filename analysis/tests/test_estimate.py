"""The estimator fits `time = t0 + work / rate` per lane; on synthetic data the
parameters recover exactly, degenerate lanes clamp instead of going negative,
and leave-one-out reports transfer gaps rather than hiding them."""

import numpy as np
import pandas as pd

from bench_analysis import estimate

MB = 2**20


def _memory(model="A", quant="q4", slope_mb=0.05, state_mb=10.0):
    return pd.DataFrame(
        [
            {
                "model": model,
                "quant": quant,
                "machine": m,
                "provider": "cpu",
                "n_ctx": c,
                "kv_mb": state_mb + slope_mb * c,
            }
            for m in ("box1", "box2")
            for c in (512, 2048, 8192)
        ]
    )


def _results(model="A", quant="q4", body_bytes=1000 * MB):
    return pd.DataFrame(
        [
            {
                "model": model,
                "quant": quant,
                "machine": m,
                "geo_body_bytes": body_bytes,
                "geo_body_params": 2 * body_bytes,
                "geo_file_bytes": body_bytes + 200 * MB,
            }
            for m in ("box1", "box2")
        ]
    )


def _probes(bw=None):
    bw = bw or {"box1": 100.0, "box2": 50.0}
    return pd.DataFrame(
        [
            {
                "machine": m,
                "provider": "cpu",
                "status": "ok",
                "kind": kind,
                "tflops": 10.0,
                "gbs": gbs,
            }
            for m in bw
            for kind, gbs in [("gemm", float("nan")), ("d2d", bw[m])]
        ]
    )


def _sweeps(t0=0.002, eta=0.8, bw=None):
    """Decode points generated exactly from the affine law."""
    bw = bw or {"box1": 100.0, "box2": 50.0}
    costs = estimate.model_costs(_results(), _memory()).iloc[0]
    rows = []
    for m, b in bw.items():
        for fill in (512, 2048, 8192):
            secs = t0 + (estimate.decode_read_mb(costs, fill) / 1024) / (eta * b)
            rows.append(
                {
                    "machine": m,
                    "provider": "cpu",
                    "model": "A",
                    "quant": "q4",
                    "kind": "decode",
                    "kv_fill": fill,
                    "tps_p50": 1.0 / secs,
                    "chunk_ms": None,
                    "tokens": None,
                }
            )
    return pd.DataFrame(rows)


def _pts(sweeps):
    return estimate.points(_results(), sweeps, _memory(), _probes())


def test_kv_fit_recovers_slope_and_state():
    fit = estimate.kv_fit(_memory(slope_mb=0.05, state_mb=10.0)).iloc[0]
    assert abs(fit.kv_slope_mb - 0.05) < 1e-9
    assert abs(fit.kv_state_mb - 10.0) < 1e-6


def test_lane_fits_recover_overhead_and_rate():
    fits = estimate.lane_fits(_pts(_sweeps(t0=0.002, eta=0.8))).set_index("machine")
    assert abs(fits.loc["box1"].t0_ms - 2.0) < 1e-6
    assert abs(fits.loc["box1"].rate - 80.0) < 1e-6  # 0.8 × 100 GB/s
    assert abs(fits.loc["box1"].eta - 0.8) < 1e-9
    assert fits.r2.min() > 1 - 1e-9


def test_overhead_dominated_lane_clamps_not_negative():
    # Constant time regardless of bytes: slope is unidentifiable — the fit
    # must degrade to t0 = mean, rate = inf, never a negative rate.
    s = _sweeps(bw={"box1": 100.0})
    s["tps_p50"] = 10.0
    fits = estimate.lane_fits(_pts(s))
    assert abs(fits.iloc[0].t0_ms - 100.0) < 1e-6
    assert np.isinf(fits.iloc[0].rate)


def test_loo_zero_error_when_parameters_transfer():
    out = estimate.loo(_pts(_sweeps(t0=0.002, eta=0.8)))
    assert len(out) == 2
    assert out.median_err.max() < 1e-9


def test_loo_reports_the_transfer_gap():
    s1 = _sweeps(eta=0.8, bw={"box1": 100.0})
    s2 = _sweeps(eta=0.4, bw={"box2": 50.0})
    out = estimate.loo(_pts(pd.concat([s1, s2]))).set_index("machine")
    assert out.loc["box1"].median_err > 0.2  # predicted too slow
    assert out.loc["box2"].median_err > 0.2  # predicted too fast


def test_degenerate_lane_borrows_but_never_lends():
    good = _sweeps(t0=0.002, eta=0.8, bw={"box1": 100.0})
    flat = _sweeps(bw={"box2": 50.0})
    flat["tps_p50"] = 10.0
    out = estimate.loo(_pts(pd.concat([good, flat]))).set_index("machine")
    # box1's pool would be only the degenerate box2 → excluded → no row.
    assert "box1" not in out.index
    # box2 borrows box1's finite parameters and its gap is reported.
    assert np.isfinite(out.loc["box2"].median_err)
