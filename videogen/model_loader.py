import ctypes
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from .config import parse_bool_setting


@dataclass
class HardwareProfile:
    cuda_available: bool
    gpu_name: str
    gpu_total_bytes: int
    gpu_free_bytes: int
    host_total_bytes: int
    host_available_bytes: int
    bf16_supported: bool
    is_rocm: bool

    def to_dict(self) -> Dict[str, Any]:
        return {
            "cuda_available": self.cuda_available,
            "gpu_name": self.gpu_name,
            "gpu_total_bytes": self.gpu_total_bytes,
            "gpu_free_bytes": self.gpu_free_bytes,
            "gpu_total_gb": round(float(self.gpu_total_bytes) / float(1024**3), 2) if self.gpu_total_bytes > 0 else 0.0,
            "gpu_free_gb": round(float(self.gpu_free_bytes) / float(1024**3), 2) if self.gpu_free_bytes > 0 else 0.0,
            "host_total_bytes": self.host_total_bytes,
            "host_available_bytes": self.host_available_bytes,
            "host_total_gb": round(float(self.host_total_bytes) / float(1024**3), 2) if self.host_total_bytes > 0 else 0.0,
            "host_available_gb": round(float(self.host_available_bytes) / float(1024**3), 2) if self.host_available_bytes > 0 else 0.0,
            "bf16_supported": self.bf16_supported,
            "is_rocm": self.is_rocm,
        }


@dataclass
class LoadPolicy:
    name: str
    task_step: str
    task_message: str
    device_map: Optional[Any] = None
    use_safetensors: Optional[bool] = None
    low_cpu_mem_usage: bool = True
    offload_state_dict: bool = True
    offload_folder: Optional[str] = None
    enable_model_cpu_offload: bool = False

    def loader_kwargs(self, dtype: Any) -> Dict[str, Any]:
        # from_pretrained 系は low_cpu_mem_usage=True を標準化して、
        # モデルロード時のホストRAMピークを抑える。
        kwargs: Dict[str, Any] = {
            "torch_dtype": dtype,
            "low_cpu_mem_usage": bool(self.low_cpu_mem_usage),
        }
        if self.offload_state_dict:
            kwargs["offload_state_dict"] = True
        if self.offload_folder:
            kwargs["offload_folder"] = str(self.offload_folder)
        if self.device_map is not None:
            kwargs["device_map"] = self.device_map
        if self.use_safetensors is not None:
            kwargs["use_safetensors"] = self.use_safetensors
        return kwargs

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "task_step": self.task_step,
            "task_message": self.task_message,
            "device_map": self.device_map,
            "use_safetensors": self.use_safetensors,
            "low_cpu_mem_usage": self.low_cpu_mem_usage,
            "offload_state_dict": self.offload_state_dict,
            "offload_folder": self.offload_folder,
            "enable_model_cpu_offload": self.enable_model_cpu_offload,
        }


def detect_host_memory_bytes() -> tuple[int, int]:
    try:
        import psutil  # type: ignore

        vm = psutil.virtual_memory()
        return int(vm.total), int(vm.available)
    except Exception:
        pass

    # Windows fallback.
    if os.name == "nt":
        try:

            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            status = MEMORYSTATUSEX()
            status.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
                return int(status.ullTotalPhys), int(status.ullAvailPhys)
        except Exception:
            pass

    # POSIX fallback.
    try:
        page_size = int(os.sysconf("SC_PAGE_SIZE"))
        total_pages = int(os.sysconf("SC_PHYS_PAGES"))
        avail_pages = int(os.sysconf("SC_AVPHYS_PAGES"))
        total = max(0, page_size * total_pages)
        avail = max(0, page_size * avail_pages)
        return total, avail
    except Exception:
        return 0, 0


def detect_hardware_profile(*, torch_module: Any, import_error: Optional[str]) -> HardwareProfile:
    host_total, host_avail = detect_host_memory_bytes()
    if import_error or torch_module is None:
        return HardwareProfile(
            cuda_available=False,
            gpu_name="",
            gpu_total_bytes=0,
            gpu_free_bytes=0,
            host_total_bytes=host_total,
            host_available_bytes=host_avail,
            bf16_supported=False,
            is_rocm=False,
        )

    cuda_available = bool(torch_module.cuda.is_available())
    gpu_name = ""
    gpu_total = 0
    gpu_free = 0
    bf16_supported = False
    if cuda_available:
        try:
            current = torch_module.cuda.current_device()
            gpu_name = str(torch_module.cuda.get_device_name(current))
        except Exception:
            gpu_name = ""
        try:
            props = torch_module.cuda.get_device_properties(0)
            gpu_total = int(getattr(props, "total_memory", 0) or 0)
        except Exception:
            gpu_total = 0
        try:
            free_bytes, _total_bytes = torch_module.cuda.mem_get_info()
            gpu_free = int(free_bytes)
        except Exception:
            gpu_free = 0
        try:
            bf16_supported = bool(torch_module.cuda.is_bf16_supported())
        except Exception:
            bf16_supported = False
    is_rocm = bool(getattr(getattr(torch_module, "version", None), "hip", None))
    return HardwareProfile(
        cuda_available=cuda_available,
        gpu_name=gpu_name,
        gpu_total_bytes=gpu_total,
        gpu_free_bytes=gpu_free,
        host_total_bytes=host_total,
        host_available_bytes=host_avail,
        bf16_supported=bf16_supported,
        is_rocm=is_rocm,
    )


def resolve_offload_dir(base_dir: Path) -> Path:
    offload_dir_raw = os.environ.get("VIDEOGEN_PRETRAINED_OFFLOAD_DIR", "").strip()
    if offload_dir_raw:
        offload_dir = Path(offload_dir_raw).expanduser()
        if not offload_dir.is_absolute():
            offload_dir = (base_dir / offload_dir).resolve()
    else:
        offload_dir = (base_dir / "tmp" / "pretrained_offload").resolve()
    offload_dir.mkdir(parents=True, exist_ok=True)
    return offload_dir


def build_load_policies(
    *,
    settings: Dict[str, Any],
    hardware: HardwareProfile,
    prefer_gpu_device_map: bool,
    offload_dir: Path,
) -> list[LoadPolicy]:
    server = settings.get("server", {})
    threshold_gb = float(server.get("vram_gpu_direct_load_threshold_gb", 48.0) or 48.0)
    threshold_bytes = int(max(1.0, threshold_gb) * (1024**3))
    allow_auto_map = parse_bool_setting(server.get("enable_device_map_auto", True), default=True)
    allow_cpu_offload = parse_bool_setting(server.get("enable_model_cpu_offload", True), default=True)
    policies: list[LoadPolicy] = []

    if prefer_gpu_device_map and hardware.cuda_available and hardware.gpu_total_bytes >= threshold_bytes:
        policies.append(
            LoadPolicy(
                name="gpu_direct_map_string",
                task_step="model_load_gpu",
                task_message="VRAMにロード中 (device_map='cuda')",
                device_map="cuda",
                offload_folder=str(offload_dir),
            )
        )

    if prefer_gpu_device_map and hardware.cuda_available and allow_auto_map:
        policies.append(
            LoadPolicy(
                name="device_map_auto",
                task_step="model_load_auto_map",
                task_message="自動device_mapでロード中",
                device_map="auto",
                offload_folder=str(offload_dir),
            )
        )

    if prefer_gpu_device_map and hardware.cuda_available and allow_cpu_offload:
        policies.append(
            LoadPolicy(
                name="cpu_offload_after_load",
                task_step="model_load_cpu_offload",
                task_message="CPUオフロード有効化中",
                device_map="auto" if allow_auto_map else None,
                offload_folder=str(offload_dir),
                enable_model_cpu_offload=True,
            )
        )

    # 最終フォールバック: CPUメモリ最適化ロード。device_map が使えない環境でも成立する。
    policies.append(
        LoadPolicy(
            name="cpu_low_mem",
            task_step="model_load_cpu",
            task_message="CPU低メモリモードでロード中",
            device_map=None,
            offload_folder=str(offload_dir),
        )
    )
    return policies


def build_load_policy_preview(
    *,
    settings: Dict[str, Any],
    hardware: HardwareProfile,
    offload_dir: Path,
) -> Dict[str, Any]:
    server = settings.get("server", {})
    policies = build_load_policies(
        settings=settings,
        hardware=hardware,
        prefer_gpu_device_map=True,
        offload_dir=offload_dir,
    )
    return {
        "vram_gpu_direct_load_threshold_gb": float(server.get("vram_gpu_direct_load_threshold_gb", 48.0) or 48.0),
        "enable_device_map_auto": parse_bool_setting(server.get("enable_device_map_auto", True), default=True),
        "enable_model_cpu_offload": parse_bool_setting(server.get("enable_model_cpu_offload", True), default=True),
        "candidates": [policy.to_dict() for policy in policies],
        "selected_policy_name": policies[0].name if policies else "",
    }
