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

# A lane lends its parameters to leave-one-out pools only when the affine model
# actually describes it — a poor fit is a lane the model doesn't capture, and
# its parameters are noise to everyone else.
LEND_R2 = 0.85


def kv_fit(memory: pd.DataFrame) -> pd.DataFrame:
    """Per (model, quant): `kv_mb = state_mb + slope_mb·n_ctx` fit over the
    pooled allocator ladder. The ladder is exact and machine-independent, so
    points are pooled across machines; a negative fitted intercept is clamped
    (pure-attention models fit through the origin, noise-negative)."""
    rows = []
    for (model, quant), g in memory.dropna(subset=["n_ctx", "kv_mb"]).groupby(
        ["model", "quant"]
    ):
        pts = g.groupby("n_ctx").kv_mb.median()
        if len(pts) < 2:
            continue
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


def points(
    results: pd.DataFrame, sweeps: pd.DataFrame, memory: pd.DataFrame, probes: pd.DataFrame
) -> pd.DataFrame:
    """One row per usable sweep point, in the affine model's coordinates:
    `y = t0 + x / rate`, where a decode point is (x = GB streamed per token,
    y = seconds per token, ceiling = probed bandwidth) and a shallow prefill
    chunk is (x = TFLOPs in the chunk, y = seconds for the chunk, ceiling =
    probed gemm TFLOP/s). One frame, two `kind`s, same geometry."""
    costs = model_costs(results, memory).set_index(["model", "quant"])
    ceil = _ceilings(probes).set_index(["machine", "provider"])
    rows = []
    for r in sweeps.itertuples():
        key_mq, key_lane = (r.model, r.quant), (r.machine, r.provider)
        if key_mq not in costs.index or key_lane not in ceil.index:
            continue
        c, lane = costs.loc[key_mq], ceil.loc[key_lane]
        if r.kind == "decode" and r.tps_p50 and r.tps_p50 > 0:
            rows.append(
                {
                    "machine": r.machine,
                    "provider": r.provider,
                    "model": r.model,
                    "quant": r.quant,
                    "kind": "decode",
                    "x": decode_read_mb(c, r.kv_fill) / 1024,
                    "y": 1.0 / r.tps_p50,
                    "ceiling": lane.bw_gbs,
                }
            )
        elif r.kind == "prefill" and r.chunk_ms and r.chunk_ms > 0 and r.tokens <= SHALLOW_TOKENS:
            rows.append(
                {
                    "machine": r.machine,
                    "provider": r.provider,
                    "model": r.model,
                    "quant": r.quant,
                    "kind": "prefill",
                    "x": 2 * c.body_params * 512 / 1e12,
                    "y": r.chunk_ms / 1e3,
                    "ceiling": lane.gemm_tflops,
                }
            )
    return pd.DataFrame(rows)


def _lane_class(provider: str) -> str:
    return "cpu" if provider == "cpu" else "gpu"


def _affine(xs: np.ndarray, ys: np.ndarray) -> tuple[float, float]:
    """Least-squares `y = t0 + x·sec_per_unit`, both parameters clamped
    non-negative: a lane whose points cannot identify the slope (an
    overhead-dominated device) degrades to `t0 = mean(y)`, slope 0, rather
    than a nonsense negative rate."""
    keep = np.isfinite(xs) & np.isfinite(ys)
    xs, ys = xs[keep], ys[keep]
    if len(xs) < 2 or np.ptp(xs) == 0:
        return float(np.mean(ys)) if len(xs) else 0.0, 0.0
    sec_per_unit, t0 = np.polyfit(xs, ys, 1)
    # Unidentifiable rate: negative, or contributing less than 0.1% of the
    # observed time across the whole x range — overhead is the whole story.
    if sec_per_unit < 0 or sec_per_unit * float(np.ptp(xs)) < 1e-3 * float(np.mean(ys)):
        return float(np.mean(ys)), 0.0
    if t0 < 0:
        return 0.0, float((xs * ys).sum() / (xs * xs).sum())
    return float(t0), float(sec_per_unit)


def lane_fits(pts: pd.DataFrame) -> pd.DataFrame:
    """Per (lane, kind): the affine fit over that lane's points — `t0_ms`
    (per-token / per-chunk overhead), `rate` (GB/s or TFLOP/s once the
    overhead is paid), `eta` (rate as a fraction of the probed ceiling), and
    the fit quality (`r2`, `n`). These two parameters are the lane; whether
    they transfer across lanes is `loo`'s verdict."""
    rows = []
    for (m, p, kind), g in pts.groupby(["machine", "provider", "kind"]):
        t0, spu = _affine(g.x.to_numpy(), g.y.to_numpy())
        pred = t0 + spu * g.x
        ss_res = float(((g.y - pred) ** 2).sum())
        ss_tot = float(((g.y - g.y.mean()) ** 2).sum())
        rate = 1 / spu if spu > 0 else float("inf")
        rows.append(
            {
                "machine": m,
                "provider": p,
                "kind": kind,
                "lane_class": _lane_class(p),
                "t0_ms": t0 * 1e3,
                "rate": rate,
                "ceiling": float(g.ceiling.iloc[0]),
                "eta": rate / float(g.ceiling.iloc[0]),
                "r2": 1 - ss_res / ss_tot if ss_tot > 0 else float("nan"),
                "n": len(g),
            }
        )
    return pd.DataFrame(rows)


def lenders(fits: pd.DataFrame) -> pd.DataFrame:
    """The lanes whose parameters are trustworthy enough to leave the lane:
    finite rate and a fit that actually describes the points (r2 ≥ LEND_R2).
    Everything pooled — LOO, class parameters, the report's headline ranges —
    draws from these; the rest borrow, never lend."""
    return fits[np.isfinite(fits.eta) & (fits.r2 >= LEND_R2)]


def loo(pts: pd.DataFrame) -> pd.DataFrame:
    """Leave-one-lane-out on the affine parameters: predict each lane's
    measured points with `t0` and `eta` pooled (median) from the *other*
    lanes of its class, that lane's own probed ceiling supplying the units.
    Per (lane, kind): median and worst absolute relative error on the
    predicted rate. The estimator transfers only as well as this says."""
    fits = lane_fits(pts)
    rows = []
    for (m, p, kind), g in pts.groupby(["machine", "provider", "kind"]):
        good = lenders(fits)
        others = good[
            (good.kind == kind)
            & (good.lane_class == _lane_class(p))
            & ~((good.machine == m) & (good.provider == p))
        ]
        if others.empty:
            continue
        t0_hat = others.t0_ms.median() / 1e3
        eta_hat = others.eta.median()
        pred = t0_hat + g.x / (eta_hat * g.ceiling)
        err = (pred / g.y - 1).abs()
        rows.append(
            {
                "machine": m,
                "provider": p,
                "kind": kind,
                "n_lanes_pooled": len(others),
                "t0_hat_ms": float(t0_hat * 1e3),
                "eta_hat": float(eta_hat),
                "median_err": float(err.median()),
                "worst_err": float(err.max()),
            }
        )
    return pd.DataFrame(rows)


def class_params(fits: pd.DataFrame) -> pd.DataFrame:
    """Per (lane_class, kind): the pooled parameters a fleet prediction uses —
    median t0 and η over the lanes good enough to lend (finite rate, fit
    r2 ≥ LEND_R2) — with the lane count behind them."""
    out = (
        lenders(fits).groupby(["lane_class", "kind"])
        .agg(t0_ms=("t0_ms", "median"), eta=("eta", "median"), n_lanes=("eta", "size"))
        .reset_index()
    )
    return out


def fleet_coefficients(
    results: pd.DataFrame, sweeps: pd.DataFrame, memory: pd.DataFrame, probes: pd.DataFrame
) -> dict:
    """Everything the fleet calculator needs, as one JSON-ready dict: per-model
    costs, per-class pooled lane parameters, and the leave-one-out error that
    must be shown next to every prediction."""
    pts = points(results, sweeps, memory, probes)
    fits = lane_fits(pts)
    cls = class_params(fits)
    errs = loo(pts)
    errs["lane_class"] = errs.provider.map(_lane_class)
    loo_by_class = (
        errs.groupby(["lane_class", "kind"]).median_err.median().rename("median_err").reset_index()
    )
    return {
        "models": [
            {
                "model": r.model,
                "quant": r.quant,
                "file_gb": r.file_bytes / 2**30,
                "body_gb": r.body_bytes / 2**30,
                "body_gparams": r.body_params / 1e9,
                "kv_mb_per_1k": r.kv_slope_mb * 1024,
                "kv_state_mb": r.kv_state_mb,
            }
            for r in model_costs(results, memory).itertuples()
        ],
        "classes": [
            {
                "lane_class": r.lane_class,
                "kind": r.kind,
                "t0_ms": r.t0_ms,
                "eta": r.eta,
                "n_lanes": int(r.n_lanes),
            }
            for r in cls.itertuples()
        ],
        "loo": [
            {"lane_class": r.lane_class, "kind": r.kind, "median_err": r.median_err}
            for r in loo_by_class.itertuples()
        ],
    }
