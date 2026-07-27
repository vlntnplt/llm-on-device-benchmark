"""Work enumeration, driven by `models.yaml`.

`models.yaml` is the single source of truth for *what to run*: per backend block
(`gguf` for `ggml`) a `repo`, a `common` glob list, and a
`quants` map keyed by the contract quant enum → that quant's HF `files`. We emit
one `Variant` per declared quant — the key *is* the wire value, no normalization.

The matrix is exactly what `models.yaml` declares, not what a filesystem scan
finds; a declared-but-unfetched quant is dropped with a loud warning, never
silently. The provider axis is the exe's call — `providers()` asks each artifact.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path

import yaml

from ._log import warn
from .config import REGISTRY, Backend

# backend key → the models.yaml block holding its artifact.
BACKEND_BLOCK = {"ggml": "gguf"}
# the contract `--quant` enum (events/results schema).
QUANTS = {"fp16", "q8", "q4", "q2"}


@dataclass(frozen=True)
class Variant:
    model: str  # the models.yaml key
    model_path: Path  # resolved artifact (--model): the .gguf file
    quant: str  # contract enum (fp16|q8|q4|q2), verbatim from models.yaml


def _matched(block_dir: Path, files: list[str]) -> list[Path]:
    """Real files under a fetched block dir matching any of a quant's `files` globs."""
    hits: set[Path] = set()
    for pat in files or []:
        hits.update(p for p in block_dir.glob(pat) if p.is_file())
    return sorted(hits)


def variants(models_dir: Path, backend_key: str) -> list[Variant]:
    """One variant per (model, quant) declared for this backend and actually fetched."""
    block = BACKEND_BLOCK.get(backend_key)
    if block is None:
        raise SystemExit(f"unknown backend {backend_key!r}; known: {list(BACKEND_BLOCK)}")
    registry = yaml.safe_load(REGISTRY.read_text()) if REGISTRY.exists() else {}

    out: list[Variant] = []
    for model, entry in sorted((registry or {}).items()):
        spec = (entry or {}).get(block)
        if not spec:
            continue  # no artifact of this backend for this model
        quants = spec.get("quants")
        if not quants:
            raise SystemExit(
                f"{model}.{block}: no `quants` map in {REGISTRY.name} "
                f"(expected e.g. quants: {{ q4: {{ files: [...] }} }})"
            )
        block_dir = models_dir / model / block
        if not block_dir.is_dir():
            warn(
                f"{model}.{block}: declared but not fetched ({block_dir}) — skipping; "
                f"run `bench fetch {model}`"
            )
            continue
        for quant, qspec in sorted(quants.items()):
            if quant not in QUANTS:
                raise SystemExit(
                    f"{model}.{block}.quants: {quant!r} is not a contract quant "
                    f"({'|'.join(sorted(QUANTS))})"
                )
            hits = _matched(block_dir, (qspec or {}).get("files", []))
            if not hits:
                warn(
                    f"{model}.{block} {quant}: no fetched files match "
                    f"{(qspec or {}).get('files')} under {block_dir} — skipping"
                )
                continue
            ggufs = [p for p in hits if p.suffix == ".gguf"]
            if len(ggufs) != 1:
                names = ", ".join(p.name for p in ggufs) or "(none)"
                warn(f"{model}.gguf {quant}: expected one .gguf, found {names} — skipping")
                continue
            out.append(Variant(model=model, model_path=ggufs[0], quant=quant))
    return out


def select(variants: list[Variant], names: list[str] | None) -> list[Variant]:
    """Restrict variants to the named models (`--model`); an unknown name is a
    loud error, not an empty run."""
    if not names:
        return variants
    known = {v.model for v in variants}
    unknown = [n for n in names if n not in known]
    if unknown:
        raise SystemExit(f"--model: unknown model(s) {unknown}; fetched here: {sorted(known)}")
    return [v for v in variants if v.model in names]


def providers(backend: Backend, model_path: Path) -> list[str]:
    """Providers this *artifact* runs on this machine — the exe decides."""
    proc = subprocess.run(
        [*backend.cmd, "providers", "--model", str(model_path)],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise SystemExit(
            f"`providers` exited {proc.returncode} for {model_path}:\n{proc.stderr.strip()}"
        )
    eps = json.loads(proc.stdout)
    if not isinstance(eps, list) or not eps:
        raise SystemExit(
            f"`providers` returned {eps!r} for {model_path}; expected a non-empty array"
        )
    return eps
