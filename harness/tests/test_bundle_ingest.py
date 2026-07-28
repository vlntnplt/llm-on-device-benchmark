"""bundle → ingest round trip, plus the checks guarding untrusted tarballs.

`bench bundle` packs what `bench run` produced; `bench ingest` receives that
file from a stranger. The round trip must land byte-equivalent results with
the hostname scrubbed; the hostile cases (path traversal, control characters,
wrong structure) must die with an actionable reason, never half-land.
"""

from __future__ import annotations

import argparse
import gzip
import io
import json
import tarfile
from pathlib import Path

import pytest
from test_aggregate import _healthy_cell, _raw

from bench import aggregate
from bench.commands.bundle import cmd_bundle, submission_name
from bench.commands.ingest import cmd_ingest


def _local_run(tmp_path: Path) -> Path:
    """A dir shaped like `bench run --out` left it: results + raw trace."""
    raw = _raw([_healthy_cell()])
    results = aggregate.build(raw)
    src = tmp_path / "run"
    src.mkdir()
    (src / "ggml-results.json").write_text(json.dumps(results))
    (src / "ggml-raw.json.gz").write_bytes(gzip.compress(json.dumps(raw).encode()))
    return src


def _bundle(src: Path, out: Path, name: str) -> Path:
    cmd_bundle(argparse.Namespace(src=src, name=name, out=out))
    return out / f"submission-{name}.tar.gz"


def _ingest(tarball: Path, published: Path, force: bool = False) -> None:
    cmd_ingest(argparse.Namespace(tarball=tarball, published_dir=published, force=force))


def test_round_trip_lands_with_host_scrubbed(tmp_path):
    tarball = _bundle(_local_run(tmp_path), tmp_path, "test-box")
    published = tmp_path / "published"
    _ingest(tarball, published)

    dest = published / "test-box"
    assert sorted(p.name for p in dest.iterdir()) == [
        "README.md", "ggml-raw.json.gz", "ggml-results.json"]
    results = json.loads((dest / "ggml-results.json").read_text())
    assert results["machine"]["host"] == "test-box"  # no hostname leaves the machine
    raw = json.loads(gzip.decompress((dest / "ggml-raw.json.gz").read_bytes()))
    assert raw["machine"]["host"] == "test-box"


def test_ingest_refuses_to_overwrite_without_force(tmp_path):
    tarball = _bundle(_local_run(tmp_path), tmp_path, "test-box")
    published = tmp_path / "published"
    _ingest(tarball, published)
    with pytest.raises(SystemExit, match="--force"):
        _ingest(tarball, published)
    _ingest(tarball, published, force=True)


def test_submission_name_drops_vendor_noise():
    assert submission_name({"cpu": "AMD Ryzen 9 9950X 16-Core Processor",
                            "gpus": ["NVIDIA GeForce RTX 5080"]}) == "ryzen-9-9950x-rtx-5080"
    assert submission_name({"cpu": "Intel(R) Core(TM) Ultra 5 125U",
                            "gpus": []}) == "core-ultra-5-125u"
    assert submission_name({"cpu": "AMD Ryzen 7 255",
                            "gpus": ["AMD Radeon 780M Graphics"]}) == "ryzen-7-255-radeon-780m"
    assert submission_name({"cpu": "", "gpus": []}) == "unknown-machine"


def test_ingest_rejects_path_traversal(tmp_path):
    evil = tmp_path / "evil.tar.gz"
    with tarfile.open(evil, "w:gz") as tar:
        info = tarfile.TarInfo("../escape.json")
        info.size = 2
        tar.addfile(info, io.BytesIO(b"{}"))
    with pytest.raises(SystemExit, match="expected exactly"):
        _ingest(evil, tmp_path / "published")


def test_ingest_rejects_unexpected_files(tmp_path):
    evil = tmp_path / "evil.tar.gz"
    with tarfile.open(evil, "w:gz") as tar:
        info = tarfile.TarInfo("box/exploit.sh")
        info.size = 2
        tar.addfile(info, io.BytesIO(b"#!"))
    with pytest.raises(SystemExit, match="unexpected file"):
        _ingest(evil, tmp_path / "published")


def test_ingest_rejects_control_characters(tmp_path):
    src = _local_run(tmp_path)
    doc = json.loads((src / "ggml-results.json").read_text())
    doc["runs"][0]["job"]["sample_completions"] = ["wipe \x1b[2J the report"]
    (src / "ggml-results.json").write_text(json.dumps(doc))
    tarball = _bundle(src, tmp_path, "ctrl-box")
    with pytest.raises(SystemExit, match="suspicious strings"):
        _ingest(tarball, tmp_path / "published")


def test_ingest_rejects_not_a_tarball(tmp_path):
    fake = tmp_path / "submission-x.tar.gz"
    fake.write_bytes(b"PK\x03\x04 this is a zip actually")
    with pytest.raises(SystemExit, match="not a gzipped tarball"):
        _ingest(fake, tmp_path / "published")
