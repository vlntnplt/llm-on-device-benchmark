"""Run one backend process, sampling memory from the outside.

A spawn is the atomic unit: one `run`, `sweep`, or `probe` invocation. The
memory sampler is attached only where its samples are consumed — the
validation-job spawns (`sample=True`); gate, sweep, and probe spawns run
unsampled and carry an empty sample series. One process loads at most one
model and runs one provider, so a sampled spawn is a single clean memory
timeline. stdout is the events object and nothing else (contract); we parse
and schema-validate it before anyone downstream trusts a number.

A failed `expect` exits nonzero but still emits a valid events object (it
carries the decoded text) — we keep that. Only missing/garbled stdout is a
hard error.
"""

from __future__ import annotations

import json
import subprocess
import tempfile
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path

import psutil

from . import schema
from .sampling import Sampler


def _kill_tree(proc: subprocess.Popen) -> None:
    """Kill the process and any children, then reap, so a backstop'd spawn leaves
    nothing behind (the backends are single-process, but be defensive)."""
    try:
        root = psutil.Process(proc.pid)
        procs = [root, *root.children(recursive=True)]
    except psutil.NoSuchProcess:
        procs = []
    for p in procs:
        try:
            p.kill()
        except psutil.NoSuchProcess:
            pass
    proc.kill()  # ensure the Popen handle itself is signalled


@dataclass
class SpawnResult:
    events: dict | None  # validated events object, or None on hard failure
    samples: list[dict]  # (wall_unix_ns, rss, vram) over the process tree
    cold: bool  # first process to touch this model file (cold page cache)
    error: str | None  # last stderr line when stdout wasn't a valid events object
    timed_out: bool = False  # killed at the harness backstop

    @property
    def healthy(self) -> bool:
        # sweep/probe events carry no expects — they are healthy by existing.
        return self.events is not None and self.events.get("healthy", True)

    @property
    def truncated(self) -> bool:
        """Soft deadline cut the in-process loop below the requested K — a signal
        the cell is slow, so the harness can stop re-spawning it."""
        return (
            self.events is not None
            and len(self.events.get("iterations") or []) < self._iters_requested
        )

    _iters_requested: int = 1


def _execute(cmd: list[str], *, backstop_s: float | None, cold: bool = False,
             iters: int = 1, sample: bool = False) -> SpawnResult:
    """Spawn, sample (job spawns only), backstop-kill if needed, parse +
    schema-validate stdout."""
    killed = False
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    with Sampler(proc.pid) if sample else nullcontext() as sampler:
        try:
            stdout, stderr = proc.communicate(timeout=backstop_s)
        except subprocess.TimeoutExpired:
            _kill_tree(proc)
            stdout, stderr = proc.communicate()
            killed = True
    samples = sampler.samples if sampler else []

    if killed:
        reason = f"killed at backstop ({backstop_s:.0f}s)"
        return SpawnResult(events=None, samples=samples, cold=cold, error=reason,
                           timed_out=True, _iters_requested=iters)

    try:
        events = json.loads(stdout)
    except json.JSONDecodeError:
        reason = (stderr.strip().splitlines() or ["no stdout"])[-1][:200]
        return SpawnResult(events=None, samples=samples, cold=cold, error=reason,
                           _iters_requested=iters)

    schema.validate_events(events, label=" ".join(cmd[:2]))
    return SpawnResult(events=events, samples=samples, cold=cold, error=None,
                       _iters_requested=iters)


def run(
    cmd_prefix: list[str],
    *,
    model_path: Path,
    quant: str,
    ep: str,
    task: dict,
    iters: int,
    cold: bool = False,
    deadline_ms: int | None = None,
    backstop_s: float | None = None,
    sample: bool = False,
) -> SpawnResult:
    """One chat-task spawn (brain-check or the job). `task` is already resolved
    (documents inlined); we hand the exe everything — it does no path/template
    resolution beyond its own tokenizer.

    `sample` attaches the memory sampler — the job spawns only; the gate's
    samples feed nothing. `deadline_ms` soft-caps the in-process loop (the exe
    stops below K); `backstop_s` is the hard floor — if even one iteration
    outlives it we kill the tree and return a timed_out result with no
    events."""
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as fh:
        json.dump(task, fh)
        task_path = fh.name

    cmd = [
        *cmd_prefix, "run",
        "--model", str(model_path),
        "--quant", quant,
        "--ep", ep,
        "--task", task_path,
        "--iters", str(iters),
        *(["--deadline-ms", str(deadline_ms)] if deadline_ms else []),
        "--out", "-",
    ]
    try:
        return _execute(cmd, backstop_s=backstop_s, cold=cold, iters=iters, sample=sample)
    finally:
        Path(task_path).unlink(missing_ok=True)


def sweep(
    cmd_prefix: list[str],
    *,
    model_path: Path,
    quant: str,
    ep: str,
    deadline_ms: int | None = None,
    backstop_s: float | None = None,
) -> SpawnResult:
    """One sweep spawn: the exe measures its prefill/decode points with adaptive
    in-process repetition. `deadline_ms` soft-caps the point loop (the first
    point of each kind always completes); `backstop_s` hard-kills a hang."""
    cmd = [
        *cmd_prefix, "sweep",
        "--model", str(model_path),
        "--quant", quant,
        "--ep", ep,
        *(["--deadline-ms", str(deadline_ms)] if deadline_ms else []),
        "--out", "-",
    ]
    return _execute(cmd, backstop_s=backstop_s)


def probe(
    cmd_prefix: list[str],
    *,
    ep: str,
    backstop_s: float | None = None,
) -> SpawnResult:
    """One device-ceiling probe spawn — no model."""
    cmd = [*cmd_prefix, "probe", "--ep", ep, "--out", "-"]
    return _execute(cmd, backstop_s=backstop_s)
