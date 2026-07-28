"""`bench bundle` — pack a local run into a single submission tarball.

The contributor-side counterpart of `bench publish`: same validation, same
README, but the output is one `submission-<name>.tar.gz` to attach to a
submission issue instead of a folder in a checkout. `<name>` derives from the
measured machine ("<cpu>-<gpu>", slugged, vendor noise dropped) — never from
free text — and `machine.host` is rewritten to it in every packed file, so a
submission carries no hostname.

`bench ingest` is the receiving end; the tarball layout is its contract:
one top-level `<name>/` directory holding `README.md`, `<backend>-results.json`,
and `<backend>-raw.json.gz`.
"""

from __future__ import annotations

import argparse
import gzip
import io
import json
import re
import tarfile
import time
import urllib.parse
from pathlib import Path

from .. import schema
from .._log import log
from ..config import PROJECT_URL
from .publish import _render_readme

_PAREN_NOISE = re.compile(r"\((?:R|TM)\)", re.IGNORECASE)
_CPU_NOISE = re.compile(r"\b(AMD|Intel|Processor|CPU)\b|\b\d+-Core\b|@.*$|\bw/.*$", re.IGNORECASE)
_GPU_NOISE = re.compile(r"\b(NVIDIA|GeForce|AMD|Graphics)\b", re.IGNORECASE)


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def submission_name(machine: dict) -> str:
    """The "<cpu>-<gpu>" submission label, from parsed machine fields only —
    e.g. "ryzen-9-9950x-rtx-5080". A cleaned label that comes out empty falls
    back to the raw slug rather than vanishing."""
    cpu_raw = machine.get("cpu") or ""
    cpu = _slug(_CPU_NOISE.sub(" ", _PAREN_NOISE.sub(" ", cpu_raw))) or _slug(cpu_raw)
    gpu_raw = (machine.get("gpus") or [""])[0]
    gpu = _slug(_GPU_NOISE.sub(" ", _PAREN_NOISE.sub(" ", gpu_raw))) or _slug(gpu_raw)
    return "-".join(part for part in (cpu, gpu) if part) or "unknown-machine"


def _add_bytes(tar: tarfile.TarFile, arcname: str, data: bytes) -> None:
    info = tarfile.TarInfo(arcname)
    info.size = len(data)
    info.mtime = int(time.time())
    tar.addfile(info, io.BytesIO(data))


def cmd_bundle(args: argparse.Namespace) -> None:
    src: Path = args.src
    results = sorted(src.glob("*-results.json"))
    if not results:
        raise SystemExit(
            f"{src}: no *-results.json to bundle — run the benchmark first "
            f"(`bench run --backend ggml --out {src}`)"
        )

    # Validate everything before writing anything, so a bad file never ships.
    docs: list[tuple[Path, dict]] = []
    for rp in results:
        doc = json.loads(rp.read_text())
        schema.validate_results(doc, label=str(rp))
        docs.append((rp, doc))

    name = args.name or submission_name(docs[0][1]["machine"])
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,63}", name):
        raise SystemExit(f"submission name {name!r} must be lowercase [a-z0-9-], ≤64 chars")

    out_path: Path = args.out / f"submission-{name}.tar.gz"
    with tarfile.open(out_path, "w:gz") as tar:
        for rp, doc in docs:
            doc["machine"]["host"] = name  # no hostnames leave the machine
            _add_bytes(tar, f"{name}/{rp.name}", json.dumps(doc, indent=2).encode())
            raw_path = src / f"{doc['backend']}-raw.json.gz"
            if raw_path.exists():
                raw = json.loads(gzip.decompress(raw_path.read_bytes()))
                raw["machine"]["host"] = name
                _add_bytes(tar, f"{name}/{raw_path.name}",
                           gzip.compress(json.dumps(raw).encode()))
        _add_bytes(tar, f"{name}/README.md", _render_readme(name, docs).encode())

    title = urllib.parse.quote(f"submission: {name}")
    log(f"bundled {name} → {out_path} ({out_path.stat().st_size // 1024} KB)")
    log("submit: open a new issue and attach this file —")
    log(f"  {PROJECT_URL}/issues/new?template=submission.yml&title={title}")
