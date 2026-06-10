"""Host identity for results.machine — host / os / cpu / gpus.

Independent of any backend: the CPU label comes from the OS, the GPU labels from
NVML. (A backend also reports its own `device` per run; that's the device a given
provider actually used, which can be a subset of what's installed here.)

`host` is the machine's name — the hostname by default, or whatever `bench run
--machine` was given. It's how the analysis loader labels a machine when results
aren't filed under a per-machine subdir, so it beats slugging the GPU.
"""

from __future__ import annotations

import platform
import re
from pathlib import Path

import psutil

from . import sampling

_OS = {"Linux": "linux", "Darwin": "macos", "Windows": "windows"}


def _cpu_label() -> str:
    if platform.system() == "Linux":
        try:
            for line in Path("/proc/cpuinfo").read_text().splitlines():
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
        except OSError:
            pass
    if platform.system() == "Darwin":
        import subprocess

        try:
            return subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True,
                text=True,
                check=True,
            ).stdout.strip()
        except Exception:  # noqa: BLE001
            pass
    return platform.processor() or platform.machine() or "unknown"


def info(name: str | None = None) -> dict:
    """Machine block for results. `name` overrides the host label (else the
    hostname); the rest is probed from the OS and NVML. `cpu_cores`/`cpu_threads`
    (physical/logical) describe the hardware the numbers were measured on."""
    return {
        "host": name or platform.node() or "unknown",
        "os": _OS.get(platform.system(), platform.system().lower()),
        "cpu": re.sub(r"\s+", " ", _cpu_label()),
        "cpu_cores": psutil.cpu_count(logical=False) or psutil.cpu_count() or 1,
        "cpu_threads": psutil.cpu_count() or 1,
        "gpus": sampling.gpu_names(),
    }
