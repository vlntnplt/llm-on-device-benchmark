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

GTT (the RSS blind spot): on a GPU backend the weights and KV cache are uploaded
into driver-allocated buffer objects that never enter the process address space,
so psutil RSS misses them entirely. On a unified-memory APU — the only place a
non-NVIDIA GPU backend runs here — those live in GTT, which is *system RAM*
apertured for the GPU (a 4 GB model showed 670 MB RSS but 4.1 GB resident GTT).
Linux exposes per-PID GTT via DRM fdinfo, so we read it and fold it into `rss`:
the reported RAM then reflects the true footprint, not the CPU-side scaffolding.
(NVIDIA goes through NVML instead — CUDA device memory is *not* system RAM, so it
belongs in VRAM, not RSS.)
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


# Read per-PID GTT off DRM fdinfo only on a non-NVIDIA Linux box with a render node
# (NVIDIA's GPU memory comes from NVML and isn't system RAM, so it must not land in
# RSS). Cheap /proc reads — no driver calls — so it can't perturb GPU decode.
_HAS_DRM = (
    platform.system() == "Linux" and not _NVML_HANDLES and any(Path("/dev/dri").glob("renderD*"))
)


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


_GTT_UNITS = {"B": 1 / 1024, "KiB": 1, "MiB": 1024, "GiB": 1024 * 1024}


def _fdinfo_kib(line: str) -> int:
    """`drm-…: <n> [B|KiB|MiB|GiB]` → KiB (the kernel emits the unit; bare = bytes)."""
    parts = line.split(":", 1)[1].split()
    if not parts:
        return 0
    unit = parts[1] if len(parts) > 1 else "B"
    return int(float(parts[0]) * _GTT_UNITS.get(unit, 1))


def _drm_gtt_bytes(pids: set[int]) -> int:
    """Summed resident GTT (GPU-pinned system RAM) across the pids' DRM fds.

    Per DRM fdinfo (Documentation/gpu/drm-usage-stats.rst). A process can hold
    several fds to the same GPU client and the counters repeat across them, so
    dedupe per (pid, drm-client-id) with max, then sum distinct clients. Prefer
    `resident` (backed right now); fall back to the older `memory`/`total` keys on
    kernels that don't emit it. fds open and close mid-run, so tolerate races."""
    total = 0
    for pid in pids:
        per_client: dict[str, int] = {}
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
            resident = memory = grand = 0
            for line in text.splitlines():
                if line.startswith("drm-client-id:"):
                    client = line.split(":", 1)[1].strip()
                elif line.startswith("drm-resident-gtt:"):
                    resident = _fdinfo_kib(line)
                elif line.startswith("drm-memory-gtt:"):
                    memory = _fdinfo_kib(line)
                elif line.startswith("drm-total-gtt:"):
                    grand = _fdinfo_kib(line)
            gtt = resident or memory or grand
            per_client[client] = max(per_client.get(client, 0), gtt)
        total += sum(per_client.values())
    return total * 1024  # KiB → bytes


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
        memory_full_info — no /proc/smaps walk. (On an APU the weights live in GTT,
        outside RSS; `_run` adds that back — see `_drm_gtt_bytes`.)"""
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
        gtt = 0
        while not self._stop.is_set():
            try:
                rss, pids = self._tree_stats()
            except (psutil.NoSuchProcess, ProcessLookupError):
                break
            # VRAM and GTT share the slow tick: per-PID NVML enumeration at
            # 100 Hz perturbs decode, and both move slowly anyway. Carry forward.
            if pids and tick % VRAM_POLL_EVERY == 0:
                if _NVML_HANDLES:
                    vram = _vram_bytes(pids)
                elif _HAS_DRM:
                    gtt = _drm_gtt_bytes(pids)
            # GTT is the GPU-pinned system RAM psutil RSS can't see (0 off the APU
            # path) — fold it in so `rss` is the true RAM footprint.
            self.samples.append({"t": time.time_ns(), "rss": rss + gtt, "vram": vram})
            tick += 1
            time.sleep(SAMPLE_INTERVAL_S)

    def __enter__(self) -> Sampler:
        self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._stop.set()
        self._thread.join()
