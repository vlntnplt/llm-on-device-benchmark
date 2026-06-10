"""Run one cell as one process, sampling memory from the outside.

A spawn is the atomic unit: `<argv> run --model … --ep … --task … --iters K`,
with the memory sampler attached for its lifetime. One process loads the model
exactly once and runs one provider, so its samples are a single clean memory
timeline. stdout is the events object and nothing else (contract); we parse
and schema-validate it before anyone downstream trusts a number.

A failed `expect` exits nonzero but still emits a valid events object (it carries
the decoded text) — we keep that. Only missing/garbled stdout is a hard error.
"""

from __future__ import annotations

import json
import subprocess
import tempfile
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
    timed_out: bool = False  # killed at the harness backstop (too slow for one iteration)

    @property
    def healthy(self) -> bool:
        return self.events is not None and self.events["healthy"]

    @property
    def truncated(self) -> bool:
        """Soft deadline cut the in-process loop below the requested K — a signal
        the cell is slow, so the harness can stop re-spawning it."""
        return self.events is not None and len(self.events["iterations"]) < self._iters_requested

    _iters_requested: int = 1


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
) -> SpawnResult:
    """Spawn one (model, variant, provider, task) cell. `task` is already resolved
    (documents inlined); we hand the exe everything — it does no path/template
    resolution beyond its own tokenizer.

    `deadline_ms` soft-caps the in-process loop (the exe stops below K);
    `backstop_s` is the hard floor — if even one iteration outlives it we kill the
    tree and return a timed_out result with no events."""
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as fh:
        json.dump(task, fh)
        task_path = fh.name

    cmd = [
        *cmd_prefix,
        "run",
        "--model",
        str(model_path),
        "--quant",
        quant,
        "--ep",
        ep,
        "--task",
        task_path,
        "--iters",
        str(iters),
        *(["--deadline-ms", str(deadline_ms)] if deadline_ms else []),
        "--out",
        "-",
    ]
    killed = False
    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        with Sampler(proc.pid) as sampler:
            try:
                stdout, stderr = proc.communicate(timeout=backstop_s)
            except subprocess.TimeoutExpired:
                _kill_tree(proc)
                stdout, stderr = proc.communicate()
                killed = True
        samples = sampler.samples
    finally:
        Path(task_path).unlink(missing_ok=True)

    if killed:
        reason = f"killed at backstop ({backstop_s:.0f}s) — too slow for one iteration"
        return SpawnResult(
            events=None,
            samples=samples,
            cold=cold,
            error=reason,
            timed_out=True,
            _iters_requested=iters,
        )

    try:
        events = json.loads(stdout)
    except json.JSONDecodeError:
        reason = (stderr.strip().splitlines() or ["no stdout"])[-1][:200]
        return SpawnResult(
            events=None, samples=samples, cold=cold, error=reason, _iters_requested=iters
        )

    schema.validate_events(events, label=f"{cmd_prefix[0]} run --ep {ep}")
    return SpawnResult(
        events=events, samples=samples, cold=cold, error=None, _iters_requested=iters
    )
