"""`bench fetch` — pull model artifacts from the Hugging Face Hub into
`models/<name>/<block>/`, driven by the root `models.yaml`.

For each model's backend block (`gguf`/`onnx`) we fetch `common` + every quant's
`files` in one snapshot (the Hub de-dupes). The globs go to
`snapshot_download(allow_patterns=…)` verbatim — edit them in `models.yaml`, not
here. `--only` restricts which backends are fetched. Everything lands under
`--models-dir` (default `models/`), untracked local data you own.
"""

from __future__ import annotations

import argparse

import yaml
from huggingface_hub import snapshot_download

from ._log import die, log
from .config import REGISTRY
from .registry import BACKEND_BLOCK

HF_PREFIX = "https://huggingface.co/"


def _repo_id(repo: str) -> str:
    """`https://huggingface.co/org/name` (or a bare `org/name`) → `org/name`."""
    return repo.removeprefix(HF_PREFIX).strip("/")


def cmd_fetch(args: argparse.Namespace) -> None:
    if not REGISTRY.exists():
        die(f"no registry at {REGISTRY}")
    registry = yaml.safe_load(REGISTRY.read_text()) or {}

    blocks = list(BACKEND_BLOCK)  # which backends to fetch
    if args.only:
        blocks = [b.strip() for b in args.only.split(",")]
        bad = [b for b in blocks if b not in BACKEND_BLOCK]
        if bad:
            die(f"--only: unknown backend(s) {bad}; choose from {list(BACKEND_BLOCK)}")

    names = args.models or list(registry)  # no names → every model in the registry
    if not names:
        die(f"{REGISTRY.name} is empty — nothing to fetch")

    dest_root = args.models_dir
    for name in names:
        entry = registry.get(name)
        if entry is None:
            known = ", ".join(registry) or "(none)"
            die(f"unknown model {name!r}. Known: {known}.")
        fetched = 0
        for backend in blocks:
            block = entry.get(BACKEND_BLOCK[backend])
            if not block:
                continue  # this model has no artifact for that backend
            if not block.get("repo"):
                die(f"{name}.{BACKEND_BLOCK[backend]}: no `repo` set in {REGISTRY.name}")
            quants = block.get("quants")
            if not quants:
                die(f"{name}.{BACKEND_BLOCK[backend]}: no `quants` map in {REGISTRY.name}")
            # Pull the shared files once plus every declared quant's weights; the Hub
            # de-dupes, so one snapshot per block covers all quants in the matrix.
            patterns = list(block.get("common", []))
            for qspec in quants.values():
                patterns += (qspec or {}).get("files", [])
            dest = dest_root / name / BACKEND_BLOCK[backend]
            log(
                f"• {name}/{BACKEND_BLOCK[backend]}: {_repo_id(block['repo'])} "
                f"({len(quants)} quant(s): {', '.join(quants)}) → {dest}"
            )
            snapshot_download(
                repo_id=_repo_id(block["repo"]),
                revision=args.revision,
                local_dir=str(dest),
                allow_patterns=patterns or None,
            )
            fetched += 1
        if not fetched:
            die(f"{name}: nothing to fetch for {blocks} — no matching blocks in {REGISTRY.name}")
        log(f"✓ {name}")
