"""The subprocess seam, pinned with a fake backend.

The argv shapes spawn.py emits ARE the CLI contract's harness side — a drift
here breaks every real backend silently. The fake exe records its argv and
prints a canned, schema-valid events object per subcommand, so these tests run
with no build and no model.
"""

from __future__ import annotations

import json
import stat
import sys
from pathlib import Path

from bench import spawn

FAKE = r'''#!/usr/bin/env python3
import json, sys, time
argv = sys.argv[1:]
out_dir = {out_dir!r}
(open(out_dir + "/argv.json", "w")).write(json.dumps(argv))
mode = argv[0]
base = {{"schema_version": "3", "backend": "ggml", "provider": "cpu:0", "device": "fake",
        "versions": {{"threads": 1}}, "anchor": {{"wall_unix_ns": 1, "mono_ns": 1}}}}
span = {{"start_ns": 0, "end_ns": 100}}
geom = {{"n_layer": 1, "n_embd": 8, "n_head": 1, "n_head_kv": 1, "n_swa": 0,
        "shared_kv_layers": 0, "n_ctx_train": 512, "n_params": 10, "file_bytes": 10,
        "context": {{"n_ctx": 256, "n_batch": 256, "n_ubatch": 256}},
        "layers": [{{"kind": "full", "window": 0}}],
        "tensors": {{"embedding": {{"params": 5, "bytes": 5}}, "body": {{"params": 5, "bytes": 5}},
                    "head": {{"params": 0, "bytes": 0}}, "tied_head": True}}, "buffers": []}}
if mode == "run":
    doc = {{**base, "mode": "run", "model": "m", "quant": "q4", "task": "t", "healthy": True,
           "load": [{{"type": "model-load", **span}}], "geometry": geom,
           "iterations": [{{"events": [
               {{"type": "prefill", "context_size": 0, "tokens_count": 4, **span}},
               {{"type": "decode", "context_size": 4, "tokens_count": 2,
                 "token_ns": [1, 2], **span}},
               {{"type": "turn-end", "completion": "hi", "expect_pass": True, **span}}]}}]}}
elif mode == "sweep":
    sweep_geom = {{**geom, "memory_points": [{{
        "n_ctx": 128, "n_batch": 128, "n_ubatch": 128,
        "buffers": [{{"name": "CPU", "model_bytes": 5, "context_bytes": 2,
                     "compute_bytes": 1}}]}}]}}
    doc = {{**base, "mode": "sweep", "model": "m", "quant": "q4",
           "load": [{{"type": "model-load", **span}}], "geometry": sweep_geom,
           "prefill_chunks": [{{"type": "prefill", "context_size": 0,
                               "tokens_count": 512, **span}}],
           "decode_points": [{{"kv_fill": 0, "tokens": 4,
                              "repeats": [{{"token_ns": [1, 2], **span}}]}}]}}
elif mode == "probe":
    doc = {{**base, "mode": "probe",
           "gemm": [{{"m": 8, "n": 8, "k": 8, "dtype": "f16", "repeats": [span]}}],
           "copy": [{{"kind": "d2d", "bytes": 8, "repeats": [span]}}]}}
elif mode == "hang":
    time.sleep(60)
    doc = {{}}
print(json.dumps(doc))
'''


def _fake_backend(tmp_path: Path) -> list[str]:
    exe = tmp_path / "fake-backend"
    exe.write_text(FAKE.format(out_dir=str(tmp_path)))
    exe.chmod(exe.stat().st_mode | stat.S_IEXEC)
    return [sys.executable, str(exe)]


def _argv(tmp_path: Path) -> list[str]:
    return json.loads((tmp_path / "argv.json").read_text())


def test_run_argv_shape_and_validation(tmp_path):
    cmd = _fake_backend(tmp_path)
    result = spawn.run(cmd, model_path=Path("/m/model.gguf"), quant="q4", ep="cpu:0",
                       task={"name": "t", "messages": []}, iters=3, deadline_ms=1000)
    assert result.events is not None and result.healthy
    assert result.samples == []  # sampling is opt-in (sample=True on the job spawns)
    argv = _argv(tmp_path)
    task_path = argv[argv.index("--task") + 1]
    assert argv[0] == "run"
    assert argv[1:7] == ["--model", "/m/model.gguf", "--quant", "q4", "--ep", "cpu:0"]
    assert argv[argv.index("--iters") + 1] == "3"
    assert argv[argv.index("--deadline-ms") + 1] == "1000"
    assert argv[-2:] == ["--out", "-"]
    assert not Path(task_path).exists()  # tempfile cleaned up


def test_run_omits_deadline_when_unset(tmp_path):
    cmd = _fake_backend(tmp_path)
    spawn.run(cmd, model_path=Path("/m/model.gguf"), quant="q4", ep="cpu:0",
              task={"name": "t", "messages": []}, iters=1)
    assert "--deadline-ms" not in _argv(tmp_path)


def test_sweep_argv_shape_and_runs_unsampled(tmp_path):
    cmd = _fake_backend(tmp_path)
    result = spawn.sweep(cmd, model_path=Path("/m/model.gguf"), quant="q4", ep="vulkan:0",
                         deadline_ms=600000)
    assert result.events is not None and result.healthy  # no expects → healthy
    assert result.events["geometry"]["memory_points"][0]["n_ctx"] == 128  # schema-checked in
    assert result.samples == []  # no sampler on a sweep — nothing consumes it
    argv = _argv(tmp_path)
    assert argv[0] == "sweep" and "--task" not in argv and "--iters" not in argv
    assert argv[argv.index("--ep") + 1] == "vulkan:0"
    assert argv[argv.index("--deadline-ms") + 1] == "600000"


def test_probe_argv_shape_and_runs_unsampled(tmp_path):
    cmd = _fake_backend(tmp_path)
    result = spawn.probe(cmd, ep="cpu:0")
    assert result.events is not None
    assert result.samples == []
    assert _argv(tmp_path) == ["probe", "--ep", "cpu:0", "--out", "-"]


def test_backstop_kills_and_reports_timeout(tmp_path):
    cmd = _fake_backend(tmp_path)
    result = spawn.probe([*cmd[:-1], cmd[-1]], ep="cpu:0", backstop_s=0.5)
    # the fake's probe path answers instantly; force the hang path via a stub
    # that ignores its subcommand
    hang = tmp_path / "hang-backend"
    hang.write_text(FAKE.format(out_dir=str(tmp_path)).replace('mode = argv[0]', 'mode = "hang"'))
    hang.chmod(hang.stat().st_mode | stat.S_IEXEC)
    result = spawn.probe([sys.executable, str(hang)], ep="cpu:0", backstop_s=0.5)
    assert result.events is None and result.timed_out
    assert "backstop" in (result.error or "")
