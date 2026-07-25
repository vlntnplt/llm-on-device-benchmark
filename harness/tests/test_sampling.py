"""Outside-in sampler internals."""

from __future__ import annotations

import pytest

from bench import sampling

# Verbatim from an i915 iGPU (Core Ultra 5 125U) mid-decode, and an amdgpu APU.
# Two DRM clients per process, only one holding the allocation.
_I915_FDINFO = """\
drm-driver:\ti915
drm-client-id:\t280
drm-pdev:\t0000:00:02.0
drm-total-system0:\t3836720 KiB
drm-shared-system0:\t0
drm-active-system0:\t3823332 KiB
drm-resident-system0:\t3835792 KiB
drm-purgeable-system0:\t7588 KiB
drm-total-stolen-local0:\t0
drm-resident-stolen-local0:\t0
"""
_AMDGPU_FDINFO = """\
drm-driver:\tamdgpu
drm-client-id:\t147
drm-pdev:\t0000:c4:00.0
drm-memory-vram:\t1984704 KiB
drm-memory-gtt: \t252368 KiB
drm-memory-cpu: \t0 KiB
amd-memory-visible-vram:\t1984704 KiB
amd-requested-vram:\t1984704 KiB
drm-shared-vram:\t0 KiB
drm-shared-gtt:\t0 KiB
"""


class _Proc:
    def __init__(self, pid: int, mem: int) -> None:
        self.pid = pid
        self.usedGpuMemory = mem


def test_vram_not_double_counted_across_compute_and_graphics(monkeypatch):
    """A Vulkan process appears in BOTH NVML lists with the same device total; CUDA
    only in compute. _vram_bytes must count each pid once (max), not sum the lists —
    else Vulkan reads ~2× (the bug that made Vulkan look like it used twice the VRAM
    of CUDA)."""
    vulkan, cuda = _Proc(100, 1_800_000_000), _Proc(200, 2_100_000_000)
    monkeypatch.setattr(sampling, "_NVML_HANDLES", ["dev0"])
    monkeypatch.setattr(
        sampling.pynvml, "nvmlDeviceGetComputeRunningProcesses", lambda h: [vulkan, cuda]
    )
    monkeypatch.setattr(
        sampling.pynvml, "nvmlDeviceGetGraphicsRunningProcesses", lambda h: [vulkan]
    )  # vulkan in graphics too

    assert sampling._vram_bytes({100}) == 1_800_000_000  # counted once, not 3.6e9
    assert sampling._vram_bytes({200}) == 2_100_000_000  # compute-only, unchanged
    assert sampling._vram_bytes({100, 200}) == 3_900_000_000


@pytest.mark.parametrize(
    ("fdinfo", "system_kib", "device_kib"),
    [(_I915_FDINFO, 3_835_792, 0), (_AMDGPU_FDINFO, 252_368, 1_984_704)],
    ids=["i915-all-in-system0", "amdgpu-apu-mostly-carve-out"],
)
def test_drm_pools_split_by_region_not_by_vendor(
    tmp_path, monkeypatch, fdinfo, system_kib, device_kib
):
    """Two disagreements between drivers, both verbatim from real hardware.

    *Region names*: matching amdgpu's `gtt` alone read 0 on i915 and dropped 3.7 GB
    of a 4.5 GB footprint off the Intel NUC's Vulkan cells.

    *Which pool the bytes land in*: the same workload put everything in system RAM
    on the Intel iGPU, but on an AMD APU 1.9 GB went to the BIOS carve-out, which
    is reserved before boot and absent from MemTotal — charging that to RSS would
    invent host memory the OS never had. Counter fallback matters too: this amdgpu
    kernel emits only `drm-memory-*`, no `drm-resident-*`."""
    fdinfo_dir = tmp_path / "1" / "fdinfo"
    fdinfo_dir.mkdir(parents=True)
    (fdinfo_dir / "3").write_text(fdinfo)
    (fdinfo_dir / "4").write_text(fdinfo)  # same client twice: dedupe, don't double
    (fdinfo_dir / "5").write_text("pos:\t0\nflags:\t0100002\n")  # a plain, non-DRM fd
    monkeypatch.setattr(sampling, "Path", lambda p: tmp_path / str(p).split("/proc/")[1])

    assert sampling._drm_bytes({1}) == (system_kib * 1024, device_kib * 1024)
