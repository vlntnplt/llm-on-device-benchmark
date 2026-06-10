"""Attribute one spawn's outside-in memory samples to the prefill/decode
windows. Pure: events + samples in, MB scalars out.

Samples line up against the exe's own event timeline via the shared wall clock:
an event's `mono_ns` maps to wall time as `anchor.wall_unix_ns + (mono_ns -
anchor.mono_ns)`.

  • prefill — peak RSS/VRAM over the prefill spans of the timed iterations: the
              prompt-ingestion working set (the whole context in one batch). Null
              when no sample landed in a prefill batch (a prefill shorter than the
              ~10 ms cadence).
  • decode  — two numbers over the decode spans. *Peak* is the high-water mark:
              what the device must fit. *Sustained* is the median: the steady
              per-token generation footprint (weights + KV + activations). They
              diverge when a transient rides into the decode window — e.g. ort's
              CoreML EP compiles at the first full-context prefill and the spike
              is still draining when decode starts, so peak reads 26 GB where the
              plateau the whole decode actually sits on is 8 GB. One number can't
              carry both facts, so we report both and let the reader pick.

There is deliberately no per-task "context" (KV) figure and no idle-weights
figure: only ggml holds an idle, preallocated KV buffer to measure (tjs
materializes KV transiently inside a forward), and the phase footprints already
carry the weights. We report the phase marks both backends genuinely have.

A `Sample` is a mapping {t: wall_unix_ns, rss, vram} — bytes; vram 0 when not
measured (the run's vram_method, decided in sampling.py, governs whether the
aggregated VRAM stat is reported or nulled).
"""

from __future__ import annotations

from statistics import median

BYTES_PER_MB = 1e6

_KEYS = (
    "prefill_rss_peak_mb",
    "prefill_vram_peak_mb",
    "decode_rss_peak_mb",
    "decode_vram_peak_mb",
    "decode_rss_sustained_mb",
    "decode_vram_sustained_mb",
)


def _to_wall(anchor: dict, mono_ns: int) -> int:
    return anchor["wall_unix_ns"] + (mono_ns - anchor["mono_ns"])


def _phase_spans(events: dict, phase: str) -> list[tuple[int, int]]:
    """Wall-time [lo, hi] spans for every `phase` event across the timed iterations
    (multi-turn tasks emit one per turn)."""
    anchor = events["anchor"]
    spans = []
    for it in events["iterations"]:
        for e in it["events"]:
            if e["type"] == phase:
                spans.append((_to_wall(anchor, e["start_ns"]), _to_wall(anchor, e["end_ns"])))
    return spans


def phase_windows(events: dict, samples: list[dict]) -> dict[str, list[dict]]:
    """Split the timed-window samples into the prefill and decode phases, using
    the events' own span stamps. A sample falls in a phase if its wall
    stamp lands inside any of that phase's spans. Either list can be empty — a
    prefill batch can be shorter than the ~10 ms sample cadence, so no tick lands in
    it; the aggregated per-phase stat is then null, never invented."""
    out: dict[str, list[dict]] = {"prefill": [], "decode": []}
    if not samples:
        return out
    for phase in out:
        spans = _phase_spans(events, phase)
        out[phase] = [s for s in samples if any(lo <= s["t"] <= hi for lo, hi in spans)]
    return out


def attribute(events: dict, samples: list[dict]) -> dict:
    """Per-spawn memory scalars in MB, over the timed iterations' own
    prefill/decode spans. None when no sample landed in a phase's spans — never
    invented; the aggregated stat then collapses to null."""
    # A live process always has non-zero RSS; rss == 0 means the read failed.
    measured = [s for s in samples if s["rss"] > 0]
    if not measured:
        return dict.fromkeys(_KEYS)

    def mb(x: float) -> float:
        return round(x / BYTES_PER_MB, 1)

    def peak(window: list[dict], field: str) -> float | None:
        return mb(max(s[field] for s in window)) if window else None

    def sustained(window: list[dict], field: str) -> float | None:
        return mb(median(s[field] for s in window)) if window else None

    # Prefill is a ramp, so only its peak is meaningful; decode is a plateau, so
    # it gets both the peak and the sustained median (see the module docstring).
    windows = phase_windows(events, measured)
    out = {}
    for field in ("rss", "vram"):
        out[f"prefill_{field}_peak_mb"] = peak(windows["prefill"], field)
        out[f"decode_{field}_peak_mb"] = peak(windows["decode"], field)
        out[f"decode_{field}_sustained_mb"] = sustained(windows["decode"], field)
    return out
