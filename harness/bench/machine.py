"""Host identity for results.machine — host / os / cpu / gpus / memory.

Independent of any backend: the CPU label comes from the OS, the GPU labels from
NVML, the memory config from dmidecode (best-effort — needs root on most boxes).
(A backend also reports its own `device` per run; that's the device a given
provider actually used, which can be a subset of what's installed here.)

The memory block matters because installed config — channels × configured MT/s —
is the source of a machine's nominal bandwidth. The CPU's spec-sheet maximum can
overstate it badly (a 125U rated for LPDDR5x-6400 running plain DDR5-4800 has
75% of the "nominal" peak before any inefficiency), so the estimator must see
what is actually in the slots.

`host` is the machine's name — the hostname by default, or whatever `bench run
--machine` was given. It's how the analysis loader labels a machine when results
aren't filed under a per-machine subdir, so it beats slugging the GPU.
"""

from __future__ import annotations

import platform
import re
import subprocess
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


_MTS = re.compile(r"(\d+)\s*MT/s")


def _dimms_linux() -> list[dict] | None:
    """Populated DIMMs off `dmidecode -t memory` — size, rated vs configured
    speed, rank, and a channel key parsed from the locator. None when dmidecode
    is unavailable or needs root (run `sudo bench run …` or accept the nulls)."""
    try:
        out = subprocess.run(["dmidecode", "-t", "memory"], capture_output=True,
                             text=True, timeout=10)
    except (OSError, subprocess.TimeoutExpired):
        return None
    if out.returncode != 0 or "Memory Device" not in out.stdout:
        return None
    return _parse_dimms(out.stdout)


def _parse_dimms(text: str) -> list[dict] | None:
    dimms = []
    for block in text.split("Memory Device")[1:]:
        fields = {}
        for line in block.splitlines():
            if ":" in line:
                key, _, value = line.strip().partition(":")
                fields[key.strip()] = value.strip()
        size = fields.get("Size", "")
        if not size or "No Module" in size:
            continue
        size_gb = float(size.split()[0]) * (1 / 1024 if "MB" in size else 1)
        rated = _MTS.search(fields.get("Speed", ""))
        configured = _MTS.search(fields.get("Configured Memory Speed", ""))
        rank = fields.get("Rank", "")
        locator = fields.get("Locator", "") + "/" + fields.get("Bank Locator", "")
        # A channel is per memory controller: "Controller0-ChannelA" and
        # "Controller1-ChannelA" are two channels, not one.
        channel = re.search(r"(?i)(?:controller[ _-]?(\d+).*)?ch(?:annel)?[ _-]?([a-z0-9])",
                            locator)
        dimms.append({
            "size_gb": size_gb,
            "rated_mts": int(rated.group(1)) if rated else None,
            "configured_mts": int(configured.group(1)) if configured else None,
            "rank": int(rank) if rank.isdigit() else None,
            "_channel": (f"{channel.group(1) or ''}/{channel.group(2).upper()}"
                         if channel else locator),
        })
    return dimms or None


def _memory() -> dict:
    """Installed memory as configured. total_gb always; the config fields are
    null when the platform tool can't say (no root, macOS unified, Windows)."""
    total_gb = round(psutil.virtual_memory().total / 2**30, 1)
    dimms = _dimms_linux() if platform.system() == "Linux" else None
    if not dimms:
        return {"total_gb": total_gb, "channels": None, "configured_mts": None,
                "rated_mts": None, "rank": None, "dimms": None}
    configured = [d["configured_mts"] for d in dimms if d["configured_mts"]]
    rated = [d["rated_mts"] for d in dimms if d["rated_mts"]]
    ranks = [d["rank"] for d in dimms if d["rank"]]
    channels = len({d["_channel"] for d in dimms})
    return {
        "total_gb": total_gb,
        "channels": channels,
        "configured_mts": min(configured) if configured else None,
        "rated_mts": max(rated) if rated else None,
        "rank": min(ranks) if ranks else None,
        "dimms": [{k: v for k, v in d.items() if k != "_channel"} for d in dimms],
    }


def info(name: str | None = None) -> dict:
    """Machine block for results. `name` overrides the host label (else the
    hostname); the rest is probed from the OS, NVML, and dmidecode.
    `cpu_cores`/`cpu_threads` (physical/logical) describe the hardware the
    numbers were measured on."""
    return {
        "host": name or platform.node() or "unknown",
        "os": _OS.get(platform.system(), platform.system().lower()),
        "cpu": re.sub(r"\s+", " ", _cpu_label()),
        "cpu_cores": psutil.cpu_count(logical=False) or psutil.cpu_count() or 1,
        "cpu_threads": psutil.cpu_count() or 1,
        "gpus": sampling.gpu_names(),
        "memory": _memory(),
    }
