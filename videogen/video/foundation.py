from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Literal


@dataclass(frozen=True)
class VideoHardwareProfile:
    gpu_name: str
    cuda_available: bool
    hip_version: str
    total_vram_gb: float
    free_vram_gb: float
    host_total_gb: float
    host_available_gb: float
    bf16_supported: bool
    safe_max_width: int
    safe_max_height: int
    safe_max_frames: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "gpu_name": self.gpu_name,
            "cuda_available": self.cuda_available,
            "hip_version": self.hip_version,
            "total_vram_gb": self.total_vram_gb,
            "free_vram_gb": self.free_vram_gb,
            "host_total_gb": self.host_total_gb,
            "host_available_gb": self.host_available_gb,
            "bf16_supported": self.bf16_supported,
            "safe_max_width": self.safe_max_width,
            "safe_max_height": self.safe_max_height,
            "safe_max_frames": self.safe_max_frames,
        }


@dataclass(frozen=True)
class VideoLoadPolicy:
    mode: Literal["full_vram", "auto_map", "cpu_offload"]
    low_cpu_mem_usage: bool
    use_safetensors: bool
    device_map: Any
    enable_cpu_offload: bool
    enable_tiling: bool
    enable_slicing: bool
    preferred_dtype: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mode": self.mode,
            "low_cpu_mem_usage": self.low_cpu_mem_usage,
            "use_safetensors": self.use_safetensors,
            "device_map": self.device_map,
            "enable_cpu_offload": self.enable_cpu_offload,
            "enable_tiling": self.enable_tiling,
            "enable_slicing": self.enable_slicing,
            "preferred_dtype": self.preferred_dtype,
        }


def derive_video_hardware_profile(*, runtime: Dict[str, Any]) -> VideoHardwareProfile:
    total_vram = float(runtime.get("total_vram_gb") or 0.0)
    free_vram = float(runtime.get("free_vram_gb") or 0.0)
    # Conservative ROCm-friendly guard rails for defaults shown on UI.
    if total_vram >= 90.0:
        safe_w, safe_h, safe_frames = 1280, 720, 192
    elif total_vram >= 48.0:
        safe_w, safe_h, safe_frames = 1024, 576, 128
    else:
        safe_w, safe_h, safe_frames = 768, 432, 96
    return VideoHardwareProfile(
        gpu_name=str(runtime.get("gpu_name") or ""),
        cuda_available=bool(runtime.get("cuda_available")),
        hip_version=str(runtime.get("torch_hip_version") or ""),
        total_vram_gb=total_vram,
        free_vram_gb=free_vram,
        host_total_gb=float((runtime.get("hardware_profile") or {}).get("host_total_gb") or 0.0),
        host_available_gb=float((runtime.get("hardware_profile") or {}).get("host_available_gb") or 0.0),
        bf16_supported=bool(runtime.get("bf16_supported")),
        safe_max_width=safe_w,
        safe_max_height=safe_h,
        safe_max_frames=safe_frames,
    )


def choose_video_load_policy(*, runtime: Dict[str, Any]) -> VideoLoadPolicy:
    vram_profile = str(runtime.get("vram_profile") or "")
    dtype = str(runtime.get("selected_dtype") or runtime.get("dtype") or "float16")
    if vram_profile == "96gb_class":
        return VideoLoadPolicy(
            mode="full_vram",
            low_cpu_mem_usage=True,
            use_safetensors=True,
            device_map={"": "cuda:0"},
            enable_cpu_offload=False,
            enable_tiling=False,
            enable_slicing=False,
            preferred_dtype=dtype,
        )
    if bool(runtime.get("enable_device_map_auto", True)):
        return VideoLoadPolicy(
            mode="auto_map",
            low_cpu_mem_usage=True,
            use_safetensors=True,
            device_map="auto",
            enable_cpu_offload=bool(runtime.get("enable_model_cpu_offload", False)),
            enable_tiling=True,
            enable_slicing=False,
            preferred_dtype=dtype,
        )
    return VideoLoadPolicy(
        mode="cpu_offload",
        low_cpu_mem_usage=True,
        use_safetensors=True,
        device_map=None,
        enable_cpu_offload=True,
        enable_tiling=True,
        enable_slicing=False,
        preferred_dtype=dtype,
    )
