"""Factor measurements into model costs × machine ceilings × stack efficiency.

The sweep measures *this* model on *this* silicon; the estimator separates the
two so either can be swapped:

- a model's **decode cost** is bytes streamed per token: the weight body plus
  the KV/state read at the current fill — the body from `geometry.tensors`,
  the KV read from a linear fit of the allocator's memory ladder
  (`kv_mb = state + slope·n_ctx`: intercept = recurrent state, slope = the
  full-attention share);
- a model's **prefill cost** is compute: ≈ 2 · body-params FLOPs per token;
- a **lane** (machine × provider) turns costs into rates through its probed
  ceilings — d2d bandwidth for decode, gemm TFLOP/s for prefill — discounted
  by an **efficiency factor** η: the fraction of the bare ceiling inference
  achieves, measured per lane and pooled per provider class.

`loo` is the honesty check: predict every lane's measured sweep points using
only η pooled from *other* lanes, and report how far off that lands. The
factors are only as transferable as that number says.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Below this prompt depth a prefill chunk is ~pure GEMM (the attention term is
# still negligible), so it is the right place to read the compute efficiency.
SHALLOW_TOKENS = 1024


def kv_fit(memory: pd.DataFrame) -> pd.DataFrame:
    """Per (model, quant): `kv_mb = state_mb + slope_mb·n_ctx` fit over the
    pooled allocator ladder. The ladder is exact and machine-independent, so
    points are pooled across machines; a negative fitted intercept is clamped
    (pure-attention models fit through the origin, noise-negative)."""
    rows = []
    for (model, quant), g in memory.groupby(["model", "quant"]):
        pts = g.groupby("n_ctx").kv_mb.median()
        slope, intercept = np.polyfit(pts.index, pts.values, 1)
        rows.append(
            {
                "model": model,
                "quant": quant,
                "kv_slope_mb": float(slope),
                "kv_state_mb": float(max(intercept, 0.0)),
            }
        )
    return pd.DataFrame(rows)


def model_costs(results: pd.DataFrame, memory: pd.DataFrame) -> pd.DataFrame:
    """Per (model, quant): the machine-independent cost card numbers —
    body bytes/params (from any run's geometry; identical across machines),
    file bytes, and the KV fit."""
    geo = (
        results.dropna(subset=["geo_body_bytes"])
        .groupby(["model", "quant"])
        .agg(
            body_bytes=("geo_body_bytes", "median"),
            body_params=("geo_body_params", "median"),
            file_bytes=("geo_file_bytes", "median"),
        )
        .reset_index()
    )
    return geo.merge(kv_fit(memory), on=["model", "quant"], how="left")


def decode_read_mb(costs: pd.Series, fill: float | np.ndarray) -> float | np.ndarray:
    """Bytes (MB) streamed per decoded token at KV fill `fill`."""
    return costs.body_bytes / 2**20 + costs.kv_state_mb + costs.kv_slope_mb * fill


def _ceilings(probes: pd.DataFrame) -> pd.DataFrame:
    """Per lane: the probed d2d bandwidth (GB/s) and best gemm TFLOP/s."""
    ok = probes[probes.status == "ok"]
    bw = ok[ok.kind == "d2d"].groupby(["machine", "provider"]).gbs.max().rename("bw_gbs")
    gemm = ok[ok.kind == "gemm"].groupby(["machine", "provider"]).tflops.max().rename("gemm_tflops")
    return pd.concat([bw, gemm], axis=1).reset_index()


def lane_points(
    results: pd.DataFrame, sweeps: pd.DataFrame, memory: pd.DataFrame, probes: pd.DataFrame
) -> pd.DataFrame:
    """One row per usable sweep point, with its implied efficiency: decode
    points carry η_decode = implied bandwidth / probed bandwidth; shallow
    prefill chunks carry η_prefill = implied FLOP rate / probed gemm."""
    costs = model_costs(results, memory).set_index(["model", "quant"])
    ceil = _ceilings(probes).set_index(["machine", "provider"])
    rows = []
    for r in sweeps.itertuples():
        key_mq, key_lane = (r.model, r.quant), (r.machine, r.provider)
        if key_mq not in costs.index or key_lane not in ceil.index:
            continue
        c, lane = costs.loc[key_mq], ceil.loc[key_lane]
        if r.kind == "decode" and r.tps_p50 and r.tps_p50 > 0:
            implied_gbs = decode_read_mb(c, r.kv_fill) * r.tps_p50 / 1024
            rows.append(
                {
                    "machine": r.machine,
                    "provider": r.provider,
                    "model": r.model,
                    "quant": r.quant,
                    "kind": "decode",
                    "x": r.kv_fill,
                    "measured": r.tps_p50,
                    "ceiling": lane.bw_gbs,
                    "eta": implied_gbs / lane.bw_gbs,
                }
            )
        elif r.kind == "prefill" and r.chunk_ms and r.chunk_ms > 0 and r.tokens <= SHALLOW_TOKENS:
            tok_s = 512 / (r.chunk_ms / 1e3)
            implied_tflops = 2 * c.body_params * tok_s / 1e12
            rows.append(
                {
                    "machine": r.machine,
                    "provider": r.provider,
                    "model": r.model,
                    "quant": r.quant,
                    "kind": "prefill",
                    "x": r.tokens,
                    "measured": tok_s,
                    "ceiling": lane.gemm_tflops,
                    "eta": implied_tflops / lane.gemm_tflops,
                }
            )
    return pd.DataFrame(rows)


def _lane_class(provider: str) -> str:
    return "cpu" if provider == "cpu" else "gpu"


def efficiency(points: pd.DataFrame) -> pd.DataFrame:
    """Per (lane, kind): the median η over that lane's models and points."""
    out = points.groupby(["machine", "provider", "kind"]).eta.median().rename("eta").reset_index()
    out["lane_class"] = out.provider.map(_lane_class)
    return out


def loo(points: pd.DataFrame) -> pd.DataFrame:
    """Leave-one-lane-out: predict each lane's measured points from η pooled
    over the *other* lanes of its class; per (lane, kind) the median and worst
    absolute relative error. The whole estimator stands or falls here."""
    eff = efficiency(points)
    rows = []
    for (m, p, kind), g in points.groupby(["machine", "provider", "kind"]):
        cls = _lane_class(p)
        others = eff[
            (eff.kind == kind)
            & (eff.lane_class == cls)
            & ~((eff.machine == m) & (eff.provider == p))
        ]
        if others.empty:
            continue
        eta_hat = others.eta.median()
        pred = g.measured / g.eta * eta_hat  # measured = eta·ceiling/cost·k
        err = (pred / g.measured - 1).abs()
        rows.append(
            {
                "machine": m,
                "provider": p,
                "kind": kind,
                "n_lanes_pooled": len(others),
                "eta_hat": float(eta_hat),
                "eta_own": float(g.eta.median()),
                "median_err": float(err.median()),
                "worst_err": float(err.max()),
            }
        )
    return pd.DataFrame(rows)
