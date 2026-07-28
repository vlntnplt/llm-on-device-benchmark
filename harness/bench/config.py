"""How to invoke a backend — the one build-system coupling.

`backends/<dir>/backend.toml` is the *only* file the harness reads to learn how to
exec a backend: `{dir}` is substituted with the backend's absolute directory and
the harness appends the CLI subcommand + flags. No hardcoded exe paths — the
`--quant` value is a `models.yaml` quant key, validated against the contract enum
(fp16|q8|q4|q2) in registry.py.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
BACKENDS_DIR = REPO / "backends"
MODELS_DIR = REPO / "models"
REGISTRY = REPO / "models.yaml"  # the model registry + fetch spec
TASKS_DIR = REPO / "tasks"
RESULTS_DIR = REPO / "results"
PROJECT_URL = "https://github.com/vlntnplt/llm-on-device-benchmark"  # the submission tracker


@dataclass(frozen=True)
class Backend:
    key: str  # matches the events object's `backend`
    name: str  # human label for reports
    cmd: list[str]  # argv prefix; the harness appends `run --model … --out -`
    dir: Path


def load_backend(key: str, backends_dir: Path = BACKENDS_DIR) -> Backend:
    """Resolve a backend key to its argv prefix via backend.toml."""
    for toml_path in sorted(backends_dir.glob("*/backend.toml")):
        cfg = tomllib.loads(toml_path.read_text())
        if cfg.get("key") == key:
            bdir = toml_path.parent
            cmd = [part.replace("{dir}", str(bdir)) for part in cfg["cmd"]]
            return Backend(key=key, name=cfg.get("name", key), cmd=cmd, dir=bdir)
    raise SystemExit(f"no backends/*/backend.toml declares key = {key!r}")
