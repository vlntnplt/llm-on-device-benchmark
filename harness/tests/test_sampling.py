"""Outside-in sampler internals."""

from __future__ import annotations

from bench import sampling


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
