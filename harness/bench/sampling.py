"""Outside-in memory sampling.

A background thread reads `(wall_unix_ns, rss, vram)` off the
measured process tree every ~10 ms while a spawn runs, and never touches the
measured process itself (psutil/NVML read /proc and the driver, not the target).
The wall clock is CLOCK_REALTIME (`time.time_ns()`), the same clock the exe stamps
its anchor with, so memory.py can line samples up against the event timeline.

VRAM is per-PID NVML on NVIDIA — which catches the GPU process whatever API it
went through (CUDA *or* Vulkan), so a Vulkan run on an NVIDIA box still gets real
numbers even though Vulkan counts as `n/a` for non-NVIDIA hardware. The run's
`vram_method` (nvml / unified / n/a) is decided in aggregate.py from platform +
whether any VRAM was actually seen.

The RSS blind spot: on a GPU backend the weights and KV cache are uploaded into
driver-allocated buffer objects that never enter the process address space, so
psutil RSS misses them entirely (a 4 GB model showed 670 MB RSS). Linux exposes
those buffers per-PID via DRM fdinfo, and they land in one of two pools — which
one decides whether they are the process's RAM or the device's:

  • **system RAM the GPU pinned** — amdgpu's `gtt`, i915/xe's `system0`. This is
    the OS's own memory, so it folds into `rss` and the reported RAM reflects the
    true footprint rather than the CPU-side scaffolding.
  • **device-local** — a discrete card's `vram`, and an APU's `vram`, which is
    the BIOS carve-out: reserved before boot and missing from MemTotal, so it is
    not the host's to charge a process for. It goes to `vram`, the same pool NVML
    fills on NVIDIA.

Drivers name regions themselves, so the split matches the region, not one
vendor's word for it. Which pool a run's bytes land in is the driver's choice and
varies by machine: on an Intel iGPU everything came back in `system0`, while an
AMD APU put 1.9 GB in the carve-out and 246 MB in `gtt`. (NVIDIA goes through
NVML instead — CUDA device memory is not system RAM either.)
"""

from __future__ import annotations

import platform
import threading
import time
from pathlib import Path

import psutil

try:
    import pynvml

    pynvml.nvmlInit()
    _NVML_HANDLES = [
        pynvml.nvmlDeviceGetHandleByIndex(i) for i in range(pynvml.nvmlDeviceGetCount())
    ]
except Exception:  # noqa: BLE001 — NVML is best-effort; absent on non-NVIDIA boxes
    _NVML_HANDLES = []

NVML_AVAILABLE = bool(_NVML_HANDLES)


# Read per-PID GPU memory off DRM fdinfo only on a non-NVIDIA Linux box with a
# render node (on NVIDIA the numbers come from NVML instead). Cheap /proc reads —
# no driver calls — so it can't perturb GPU decode.
_HAS_DRM = (
    platform.system() == "Linux" and not _NVML_HANDLES and any(Path("/dev/dri").glob("renderD*"))
)
DRM_AVAILABLE = _HAS_DRM


def gpu_names() -> list[str]:
    """Human GPU labels for results.machine — NVML on NVIDIA, system_profiler on Mac."""
    names: list[str] = []
    for handle in _NVML_HANDLES:
        try:
            name = pynvml.nvmlDeviceGetName(handle)
            names.append(name.decode() if isinstance(name, bytes) else name)
        except Exception:  # noqa: BLE001
            pass
    if names:
        return names
    if platform.system() == "Darwin":
        names = _apple_gpu_names()
    return names


def _apple_gpu_names() -> list[str]:
    import json
    import subprocess

    try:
        out = subprocess.run(
            ["system_profiler", "SPDisplaysDataType", "-json"],
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        )
        cards = json.loads(out.stdout).get("SPDisplaysDataType", [])
        return [c["sppci_model"] for c in cards if c.get("sppci_model")]
    except Exception:  # noqa: BLE001
        pass
    try:
        chip = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        if chip.startswith("Apple"):
            return [chip]
    except Exception:  # noqa: BLE001
        pass
    return []


SAMPLE_INTERVAL_S = 0.01
# RSS off /proc is cheap → full ~10 ms cadence. NVML per-PID enumeration at
# 100 Hz perturbs GPU decode (corrupting the tok/s we measure), and VRAM moves
# slowly anyway, so poll it every Nth tick (~50 ms) and carry the last value.
VRAM_POLL_EVERY = 5


def _vram_bytes(pids: set[int]) -> int:
    """Summed per-PID VRAM across all NVIDIA devices for the given pids.

    A process can appear in BOTH the compute and graphics running-process lists —
    Vulkan uses both queue types, so a Vulkan run shows up in each — and
    `usedGpuMemory` is the process's *device total*, identical in both. Summing
    across the two queries therefore double-counts it (~2× on Vulkan, while CUDA —
    compute-list only — is unaffected). Dedupe per (device, pid) with max, then sum
    distinct pids."""
    total = 0
    for handle in _NVML_HANDLES:
        per_pid: dict[int, int] = {}
        for query in (
            pynvml.nvmlDeviceGetComputeRunningProcesses,
            pynvml.nvmlDeviceGetGraphicsRunningProcesses,
        ):
            try:
                for proc in query(handle):
                    if proc.pid in pids and proc.usedGpuMemory:
                        per_pid[proc.pid] = max(per_pid.get(proc.pid, 0), proc.usedGpuMemory)
            except Exception:  # noqa: BLE001
                pass
        total += sum(per_pid.values())
    return total


_FDINFO_UNITS = {"B": 1 / 1024, "KiB": 1, "MiB": 1024, "GiB": 1024 * 1024}
# Counters in preference order: `resident` is what's backed right now; the older
# `memory`/`total` keys stand in on kernels that don't emit it.
_FDINFO_KINDS = ("resident", "memory", "total")


def _fdinfo_kib(line: str) -> int:
    """`drm-…: <n> [B|KiB|MiB|GiB]` → KiB (the kernel emits the unit; bare = bytes)."""
    parts = line.split(":", 1)[1].split()
    if not parts:
        return 0
    unit = parts[1] if len(parts) > 1 else "B"
    return int(float(parts[0]) * _FDINFO_UNITS.get(unit, 1))


def _region_pool(region: str) -> str | None:
    """Which pool a DRM memory region belongs to: "system", "device", or neither.

    Region names are the driver's to choose (drm-usage-stats.rst fixes the key
    shape, not the vocabulary), and the split matters because the two pools are
    different resources:

    - **system** — RAM the GPU pinned out of the OS's own memory, invisible to
      RSS: amdgpu's `gtt`, i915/xe's `system0` and `stolen-system0`.
    - **device** — memory that is not the OS's to allocate: a discrete card's
      `vram`/`local0`, and an APU's `vram`, which is the BIOS carve-out. The
      carve-out is *reserved before boot* and absent from MemTotal (a 16 GB
      Ryzen 7 255 laptop with a 2 GB carve-out reports 13.7 GB), so folding it
      into RSS would charge a process for memory the OS never had.

    amdgpu's `cpu` region and its byte-scale hardware regions (`gds`, `gws`,
    `oa`, `doorbell`, `mmioremap`) are neither — they aren't RAM footprints."""
    if region == "gtt" or region.startswith(("system", "stolen-system")):
        return "system"
    if region.startswith(("vram", "local", "stolen-local")):
        return "device"
    return None


def _drm_bytes(pids: set[int]) -> tuple[int, int]:
    """(GPU-pinned system RAM, device-local memory) across the pids' DRM fds.

    Per DRM fdinfo (Documentation/gpu/drm-usage-stats.rst). A process can hold
    several fds to the same GPU client and the counters repeat across them, so
    dedupe per (pid, drm-client-id) with max, then sum distinct clients. A driver
    may split a pool over several regions, so sum within a counter before falling
    back to the next — and pick that counter per pool, since a kernel emitting
    `resident` for one may only emit `memory` for the other. Both pools come off
    one pass: this runs on the sample tick, so the fds are read once, not twice.
    fds open and close mid-run, so tolerate races."""
    totals = {"system": 0, "device": 0}
    for pid in pids:
        per_client: dict[str, dict[str, int]] = {}
        try:
            fds = list(Path(f"/proc/{pid}/fdinfo").iterdir())
        except OSError:
            continue
        for fd in fds:
            try:
                text = fd.read_text()
            except OSError:  # fd closed mid-read or not a readable fdinfo
                continue
            if "drm-driver:" not in text:
                continue
            client = ""
            by_pool = {p: dict.fromkeys(_FDINFO_KINDS, 0) for p in totals}
            for line in text.splitlines():
                key = line.split(":", 1)[0]
                if key == "drm-client-id":
                    client = line.split(":", 1)[1].strip()
                    continue
                for kind in _FDINFO_KINDS:
                    prefix = f"drm-{kind}-"
                    if not key.startswith(prefix):
                        continue
                    if pool := _region_pool(key[len(prefix) :]):
                        by_pool[pool][kind] += _fdinfo_kib(line)
                    break
            seen = per_client.setdefault(client, dict.fromkeys(totals, 0))
            for pool, by_kind in by_pool.items():
                best = next((by_kind[k] for k in _FDINFO_KINDS if by_kind[k]), 0)
                seen[pool] = max(seen[pool], best)
        for seen in per_client.values():
            for pool, value in seen.items():
                totals[pool] += value
    return totals["system"] * 1024, totals["device"] * 1024  # KiB → bytes


class Sampler:
    """Samples a pid's process tree in a background thread until stopped."""

    def __init__(self, pid: int) -> None:
        self._root = psutil.Process(pid)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self.samples: list[dict] = []

    def _tree_stats(self) -> tuple[int, set[int]]:
        """Summed RSS over the live process tree, plus the live pid set.

        RSS (memory_info) on every platform — it's the whole resident footprint,
        including the page-cache-backed pages that ggml's mmap'd weights live in.
        We deliberately don't track USS: it *excludes* those mmap'd weights, so it
        undercounts the very footprint we care about on the mmap backend, while on
        tjs (private weights) it just equals RSS. memory_info is also cheaper than
        memory_full_info — no /proc/smaps walk. (On an APU the weights live in the
        GPU aperture, outside RSS; `_run` adds that back — see
        `_drm_system_bytes`.)"""
        try:
            tree = [self._root, *self._root.children(recursive=True)]
        except (psutil.NoSuchProcess, ProcessLookupError):
            tree = [self._root]
        rss = 0
        pids: set[int] = set()
        for proc in tree:
            try:
                rss += proc.memory_info().rss
                pids.add(proc.pid)
            except (psutil.NoSuchProcess, psutil.AccessDenied, ProcessLookupError):
                continue
        return rss, pids

    def _run(self) -> None:
        tick = 0
        vram = 0
        pinned = 0
        while not self._stop.is_set():
            try:
                rss, pids = self._tree_stats()
            except (psutil.NoSuchProcess, ProcessLookupError):
                break
            # VRAM and the GPU aperture share the slow tick: per-PID NVML
            # enumeration at 100 Hz perturbs decode, and both move slowly
            # anyway. Carry forward.
            if pids and tick % VRAM_POLL_EVERY == 0:
                if _NVML_HANDLES:
                    vram = _vram_bytes(pids)
                elif _HAS_DRM:
                    pinned, vram = _drm_bytes(pids)
            # `pinned` is the GPU-pinned system RAM psutil RSS can't see (0 off the
            # APU path) — fold it in so `rss` is the true RAM footprint.
            self.samples.append({"t": time.time_ns(), "rss": rss + pinned, "vram": vram})
            tick += 1
            time.sleep(SAMPLE_INTERVAL_S)

    def __enter__(self) -> Sampler:
        self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._stop.set()
        self._thread.join()
