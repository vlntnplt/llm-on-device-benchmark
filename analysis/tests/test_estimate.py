"""The estimator factors measurements into cost × ceiling × efficiency; on
synthetic, perfectly bandwidth-bound data the factors recover exactly and
leave-one-out error is zero."""

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


def _probes(bw={"box1": 100.0, "box2": 50.0}):
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
            for m, b in bw.items()
            for kind, gbs in [("gemm", float("nan")), ("d2d", b)]
        ]
    )


def _sweeps(eta=0.8, bw={"box1": 100.0, "box2": 50.0}):
    """Decode points whose tps is exactly eta·bw / bytes-per-token."""
    costs = estimate.model_costs(_results(), _memory()).iloc[0]
    rows = []
    for m, b in bw.items():
        for fill in (512, 4096):
            tps = eta * b * 1024 / estimate.decode_read_mb(costs, fill)
            rows.append(
                {
                    "machine": m,
                    "provider": "cpu",
                    "model": "A",
                    "quant": "q4",
                    "kind": "decode",
                    "kv_fill": fill,
                    "tps_p50": tps,
                    "chunk_ms": None,
                    "tokens": None,
                }
            )
    return pd.DataFrame(rows)


def test_kv_fit_recovers_slope_and_state():
    fit = estimate.kv_fit(_memory(slope_mb=0.05, state_mb=10.0)).iloc[0]
    assert abs(fit.kv_slope_mb - 0.05) < 1e-9
    assert abs(fit.kv_state_mb - 10.0) < 1e-6


def test_efficiency_recovered_exactly():
    pts = estimate.lane_points(_results(), _sweeps(eta=0.8), _memory(), _probes())
    eff = estimate.efficiency(pts)
    assert len(eff) == 2
    assert (eff.eta - 0.8).abs().max() < 1e-9


def test_loo_zero_error_when_eta_transfers():
    pts = estimate.lane_points(_results(), _sweeps(eta=0.8), _memory(), _probes())
    out = estimate.loo(pts)
    assert len(out) == 2
    assert out.median_err.max() < 1e-9
    assert (out.n_lanes_pooled == 1).all()


def test_loo_reports_the_transfer_gap():
    # box2's stack only achieves half of box1's efficiency: predicting either
    # from the other must be off by the ratio, not hidden.
    s1 = _sweeps(eta=0.8, bw={"box1": 100.0})
    s2 = _sweeps(eta=0.4, bw={"box2": 50.0})
    pts = estimate.lane_points(_results(), pd.concat([s1, s2]), _memory(), _probes())
    out = estimate.loo(pts).set_index("machine")
    assert abs(out.loc["box1"].median_err - 0.5) < 1e-6  # predicted at half speed
    assert abs(out.loc["box2"].median_err - 1.0) < 1e-6  # predicted at double
