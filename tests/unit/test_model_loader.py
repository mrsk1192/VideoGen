from pathlib import Path

import pytest

from videogen.model_loader import HardwareProfile, build_load_policies

pytestmark = pytest.mark.unit


def base_settings() -> dict:
    return {
        "server": {
            "vram_gpu_direct_load_threshold_gb": 48,
            "enable_device_map_auto": True,
            "enable_model_cpu_offload": True,
        }
    }


def test_build_load_policies_prefers_gpu_direct_for_high_vram() -> None:
    hardware = HardwareProfile(
        cuda_available=True,
        gpu_name="AMD Radeon",
        gpu_total_bytes=64 * 1024**3,
        gpu_free_bytes=48 * 1024**3,
        host_total_bytes=128 * 1024**3,
        host_available_bytes=96 * 1024**3,
        bf16_supported=True,
        is_rocm=True,
    )
    policies = build_load_policies(
        settings=base_settings(),
        hardware=hardware,
        prefer_gpu_device_map=True,
        offload_dir=Path("tmp"),
    )
    assert policies
    assert policies[0].name == "gpu_direct_map_dict"
    assert policies[0].device_map == {"": "cuda"}
    kwargs = policies[0].loader_kwargs(dtype="float16")
    assert kwargs["low_cpu_mem_usage"] is True


def test_build_load_policies_prefers_auto_and_offload_for_low_vram() -> None:
    hardware = HardwareProfile(
        cuda_available=True,
        gpu_name="AMD Radeon",
        gpu_total_bytes=12 * 1024**3,
        gpu_free_bytes=6 * 1024**3,
        host_total_bytes=64 * 1024**3,
        host_available_bytes=32 * 1024**3,
        bf16_supported=True,
        is_rocm=True,
    )
    policies = build_load_policies(
        settings=base_settings(),
        hardware=hardware,
        prefer_gpu_device_map=True,
        offload_dir=Path("tmp"),
    )
    names = [p.name for p in policies]
    assert names[0] == "device_map_auto"
    assert "cpu_offload_after_load" in names
    assert "cpu_low_mem" in names


def test_build_load_policies_cpu_only_falls_back_to_cpu_low_mem() -> None:
    hardware = HardwareProfile(
        cuda_available=False,
        gpu_name="",
        gpu_total_bytes=0,
        gpu_free_bytes=0,
        host_total_bytes=32 * 1024**3,
        host_available_bytes=16 * 1024**3,
        bf16_supported=False,
        is_rocm=False,
    )
    policies = build_load_policies(
        settings=base_settings(),
        hardware=hardware,
        prefer_gpu_device_map=True,
        offload_dir=Path("tmp"),
    )
    assert policies
    assert policies[0].name == "cpu_low_mem"
