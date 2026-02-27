import os
from pathlib import Path
from typing import Any, Dict, Optional

from .config import load_raw_settings_file, parse_bool_setting

DEFAULT_ALLOC_CONF = "max_split_size_mb:128,garbage_collection_threshold:0.8,expandable_segments:True"


def apply_pre_torch_env(base_dir: Path) -> Dict[str, Any]:
    settings_path = base_dir / "data" / "settings.json"
    settings_payload = load_raw_settings_file(settings_path) or {}
    server_payload = settings_payload.get("server", {}) if isinstance(settings_payload, dict) else {}
    rocm_flag = parse_bool_setting(server_payload.get("rocm_aotriton_experimental", True), default=True)

    # Keep allocator settings consistent even when uvicorn is started directly.
    if not os.environ.get("PYTORCH_ALLOC_CONF") and not os.environ.get("PYTORCH_CUDA_ALLOC_CONF"):
        os.environ["PYTORCH_ALLOC_CONF"] = DEFAULT_ALLOC_CONF
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = DEFAULT_ALLOC_CONF
    elif os.environ.get("PYTORCH_ALLOC_CONF") and not os.environ.get("PYTORCH_CUDA_ALLOC_CONF"):
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = os.environ["PYTORCH_ALLOC_CONF"]
    elif os.environ.get("PYTORCH_CUDA_ALLOC_CONF") and not os.environ.get("PYTORCH_ALLOC_CONF"):
        os.environ["PYTORCH_ALLOC_CONF"] = os.environ["PYTORCH_CUDA_ALLOC_CONF"]

    # start.bat already sets this. For direct uvicorn starts, set from persisted settings.
    has_env = bool(str(os.environ.get("TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL", "")).strip())
    if not has_env:
        os.environ["TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL"] = "1" if rocm_flag else "0"
    return {
        "settings_file": str(settings_path),
        "aotriton_from_settings": rocm_flag,
        "aotriton_env_before": has_env,
        "aotriton_env_effective": os.environ.get("TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL", ""),
        "pytorch_alloc_conf": os.environ.get("PYTORCH_ALLOC_CONF", ""),
    }


def select_device_and_dtype(
    *,
    settings: Dict[str, Any],
    torch_module: Any,
    import_error: Optional[str],
) -> tuple[str, Any, str]:
    if import_error:
        raise RuntimeError(f"Diffusers runtime is not available: {import_error}")

    server_settings = settings.get("server", {})
    require_gpu = parse_bool_setting(server_settings.get("require_gpu", True), default=True)
    allow_cpu_fallback = parse_bool_setting(server_settings.get("allow_cpu_fallback", False), default=False)
    preferred_dtype = str(server_settings.get("preferred_dtype", "float16")).strip().lower()
    if preferred_dtype not in {"float16", "bf16"}:
        preferred_dtype = "float16"

    cuda_available = bool(torch_module.cuda.is_available())
    if not cuda_available:
        if require_gpu and not allow_cpu_fallback:
            raise RuntimeError(
                "GPU is unavailable (torch.cuda.is_available() is false) and CPU fallback is disabled in settings."
            )
        return "cpu", torch_module.float32, "float32"

    if preferred_dtype == "bf16":
        is_bf16_supported = False
        try:
            is_bf16_supported = bool(torch_module.cuda.is_bf16_supported())
        except Exception:
            is_bf16_supported = False
        if is_bf16_supported:
            return "cuda", torch_module.bfloat16, "bf16"
    return "cuda", torch_module.float16, "float16"


def runtime_diagnostics(
    *,
    settings: Dict[str, Any],
    torch_module: Any,
    import_error: Optional[str],
    diffusers_error: Optional[str],
    npu_available: bool,
    npu_backend: str,
    npu_reason: str,
    t2v_backend_default: str,
    t2v_npu_runner_configured: bool,
) -> Dict[str, Any]:
    server_settings = settings.get("server", {})
    output: Dict[str, Any] = {
        "device": "cpu",
        "diffusers_ready": False,
        "cuda_available": False,
        "rocm_available": False,
        "npu_available": bool(npu_available),
        "npu_backend": npu_backend,
        "npu_reason": npu_reason,
        "t2v_backend_default": t2v_backend_default,
        "t2v_npu_runner_configured": bool(t2v_npu_runner_configured),
        "rocm_aotriton_env": os.environ.get("TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL", ""),
        "pytorch_alloc_conf": os.environ.get("PYTORCH_ALLOC_CONF", ""),
        "require_gpu": parse_bool_setting(server_settings.get("require_gpu", True), default=True),
        "allow_cpu_fallback": parse_bool_setting(server_settings.get("allow_cpu_fallback", False), default=False),
        "preferred_dtype": str(server_settings.get("preferred_dtype", "float16")).strip().lower(),
        "allow_software_video_fallback": parse_bool_setting(
            server_settings.get("allow_software_video_fallback", False), default=False
        ),
    }
    if import_error:
        output["import_error"] = import_error
        return output

    output["cuda_available"] = bool(torch_module.cuda.is_available())
    output["rocm_available"] = bool(getattr(torch_module.version, "hip", None))
    output["torch_version"] = getattr(torch_module, "__version__", "")
    output["torch_hip_version"] = getattr(getattr(torch_module, "version", None), "hip", None)
    output["diffusers_ready"] = diffusers_error is None
    if diffusers_error:
        output["import_error"] = diffusers_error

    try:
        selected_device, _, selected_dtype = select_device_and_dtype(
            settings=settings,
            torch_module=torch_module,
            import_error=import_error or diffusers_error,
        )
        output["device"] = selected_device
        output["dtype"] = selected_dtype
    except Exception as exc:
        output["device_policy_error"] = str(exc)

    try:
        if torch_module.cuda.is_available():
            free_bytes, total_bytes = torch_module.cuda.mem_get_info()
            output["gpu_free_bytes"] = int(free_bytes)
            output["gpu_total_bytes"] = int(total_bytes)
    except Exception:
        pass

    try:
        output["bf16_supported"] = bool(torch_module.cuda.is_bf16_supported()) if torch_module.cuda.is_available() else False
    except Exception:
        output["bf16_supported"] = False
    return output

