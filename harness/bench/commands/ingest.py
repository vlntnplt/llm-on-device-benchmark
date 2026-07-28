"""`bench ingest` — validate a submission tarball and land it in results/published.

The receiving end of `bench bundle`, run by the maintainer (or the submission
workflow) on a file a stranger produced — so the tarball is untrusted input and
every check fails with a reason a human can act on:

- the archive must be exactly one `<name>/` directory of known, sane-named
  regular files within size caps — structure is checked before a byte is
  extracted, and members are read individually, never `extractall`;
- every results file must validate against `results.schema.json` at the
  current `schema_version`;
- every string in every file must be free of control characters and absurd
  lengths — these strings render into the published report;
- the raw trace's `llama_cpp_commit` is compared to the backend's pinned
  commit (a mismatch warns: it flags a submission from an unknown build).

Nothing lands unless everything passes; `--force` only overwrites an existing
submission of the same name, it never bypasses a check.
"""

from __future__ import annotations

import argparse
import gzip
import json
import re
import tarfile
from collections import Counter
from pathlib import Path

from .. import schema
from .._log import log
from ..config import BACKENDS_DIR

MAX_TARBALL_BYTES = 25 << 20  # GitHub's issue-attachment cap; real bundles are ~300 KB
MAX_MEMBER_BYTES = 64 << 20  # decompressed; a raw trace is well under 1 MB
MAX_STRING_CHARS = 16384

_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9-]{0,63}$")
_FILE_RE = re.compile(r"^(README\.md|[a-z0-9_]+-results\.json|[a-z0-9_]+-raw\.json\.gz)$")
# Control characters except \n \r \t — nothing a completion legitimately needs.
_CTRL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def _string_offences(obj: object, path: str = "$"):
    """Yield "<json-path>: <problem>" for every suspicious string, anywhere."""
    if isinstance(obj, str):
        if len(obj) > MAX_STRING_CHARS:
            yield f"{path}: string of {len(obj)} chars (cap {MAX_STRING_CHARS})"
        if _CTRL_RE.search(obj):
            yield f"{path}: control characters"
    elif isinstance(obj, dict):
        for key, value in obj.items():
            yield from _string_offences(key, f"{path}.{key}")
            yield from _string_offences(value, f"{path}.{key}")
    elif isinstance(obj, list):
        for i, value in enumerate(obj):
            yield from _string_offences(value, f"{path}[{i}]")


def _members(tar: tarfile.TarFile) -> dict[str, tarfile.TarInfo]:
    """Check the archive's structure and return {filename: member} — exactly one
    sane-named top-level dir of known regular files, or a loud reason why not."""
    out: dict[str, tarfile.TarInfo] = {}
    names = set()
    for member in tar.getmembers():
        if member.isdir():
            continue
        if not member.isreg():
            raise SystemExit(f"{member.name}: not a regular file (links/devices are rejected)")
        parts = Path(member.name).parts
        if len(parts) != 2 or ".." in parts or parts[0].startswith("/"):
            raise SystemExit(
                f"{member.name}: expected exactly <name>/<file> — is this a bundle "
                "produced by `bench bundle` / run.sh?"
            )
        top, filename = parts
        if not _NAME_RE.fullmatch(top):
            raise SystemExit(f"submission name {top!r} must be lowercase [a-z0-9-], ≤64 chars")
        if not _FILE_RE.fullmatch(filename):
            raise SystemExit(f"{member.name}: unexpected file in a submission bundle")
        if member.size > MAX_MEMBER_BYTES:
            raise SystemExit(f"{member.name}: {member.size} bytes exceeds the member cap")
        names.add(top)
        out[filename] = member
    if len(names) != 1:
        raise SystemExit(f"expected one submission directory, found {sorted(names) or 'none'}")
    if not any(f.endswith("-results.json") for f in out):
        raise SystemExit("bundle holds no *-results.json — nothing to ingest")
    return out


def _pinned_commit() -> str | None:
    """The llama.cpp commit the ggml backend pins, straight from its CMakeLists."""
    cml = BACKENDS_DIR / "ggml" / "CMakeLists.txt"
    if not cml.exists():
        return None
    match = re.search(r'LLAMACPP_GIT_TAG\s+"([0-9a-f]{7,40})"', cml.read_text())
    return match.group(1) if match else None


def cmd_ingest(args: argparse.Namespace) -> None:
    tarball: Path = args.tarball
    if not tarball.exists():
        raise SystemExit(f"{tarball}: no such file")
    if tarball.stat().st_size > MAX_TARBALL_BYTES:
        raise SystemExit(
            f"{tarball}: {tarball.stat().st_size} bytes — a real bundle is ~300 KB; refusing"
        )

    try:
        tar = tarfile.open(tarball, "r:gz")
    except tarfile.TarError as err:
        raise SystemExit(
            f"{tarball}: not a gzipped tarball ({err}) — expected the "
            "submission-<name>.tar.gz that run.sh / `bench bundle` produced"
        ) from err

    with tar:
        members = _members(tar)
        name = Path(next(iter(members.values())).name).parts[0]
        blobs = {fn: tar.extractfile(m).read() for fn, m in members.items()}

    docs: dict[str, dict] = {}
    for filename, blob in blobs.items():
        if filename == "README.md":
            continue
        payload = gzip.decompress(blob) if filename.endswith(".gz") else blob
        try:
            doc = json.loads(payload)
        except (json.JSONDecodeError, gzip.BadGzipFile) as err:
            raise SystemExit(f"{name}/{filename}: not valid JSON ({err})") from err
        offences = list(_string_offences(doc))
        if offences:
            listed = "\n".join(f"  • {o}" for o in offences[:8])
            raise SystemExit(f"{name}/{filename}: suspicious strings — rejecting:\n{listed}")
        docs[filename] = doc

    for filename, doc in docs.items():
        if not filename.endswith("-results.json"):
            continue
        version = doc.get("schema_version")
        if version != "3":
            raise SystemExit(
                f"{name}/{filename}: results schema_version={version!r}, expected '3' — "
                "the submission was built from a different release; ask for a re-run "
                "with the current one"
            )
        schema.validate_results(doc, label=f"{name}/{filename}")

    pin = _pinned_commit()
    for filename, doc in docs.items():
        if not filename.endswith("-raw.json.gz"):
            continue
        versions = next(
            (t["events"]["versions"] for cell in doc.get("cells") or []
             for t in cell.get("gate_spawns") or [] if t.get("events")), None)
        if pin and versions and not pin.startswith(versions.get("llama_cpp_commit", "")):
            log(f"⚠️  {filename}: llama_cpp_commit {versions.get('llama_cpp_commit')!r} "
                f"differs from the pinned {pin[:12]!r} — check it came from a known release")

    dest: Path = args.published_dir / name
    if dest.exists() and not args.force:
        raise SystemExit(f"{dest} already exists — pass --force to overwrite")
    dest.mkdir(parents=True, exist_ok=True)
    for filename, blob in blobs.items():
        (dest / filename).write_bytes(blob)

    results_docs = [d for f, d in docs.items() if f.endswith("-results.json")]
    machine = results_docs[0]["machine"]
    log(f"ingested {name} → {dest}  ({len(blobs)} files)")
    log(f"  machine: {machine['cpu']} | {', '.join(machine.get('gpus') or []) or 'no GPU'} | "
        f"{machine['os']}")
    for doc in results_docs:
        statuses = Counter(
            run["job"]["status"] if run["healthy"] else "unhealthy" for run in doc["runs"])
        lanes = sorted({run["provider"] for run in doc["runs"]})
        log(f"  {doc['backend']}: {len(doc['runs'])} runs on {lanes}; "
            + ", ".join(f"{status}×{n}" for status, n in sorted(statuses.items())))
    log(f"  next: rebuild the report, review, commit as `submission({name}): …`")
