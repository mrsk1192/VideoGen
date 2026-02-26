import copy
import contextlib
import dataclasses
import importlib.util
import logging
import inspect
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import traceback
import uuid
from collections import deque
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Dict, Literal, Optional
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode
from urllib.request import Request as UrlRequest, urlopen

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from huggingface_hub import HfApi, snapshot_download
from pydantic import BaseModel, Field

TORCH_IMPORT_ERROR: Optional[str] = None
DIFFUSERS_IMPORT_ERROR: Optional[str] = None
try:
    import torch
except Exception as exc:  # pragma: no cover
    TORCH_IMPORT_ERROR = str(exc)
try:
    from PIL import Image
except Exception as exc:  # pragma: no cover
    if TORCH_IMPORT_ERROR is None:
        TORCH_IMPORT_ERROR = str(exc)
DIFFUSERS_COMPONENTS: Dict[str, Any] = {}


BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
DATA_DIR = BASE_DIR / "data"

DEFAULT_SETTINGS: Dict[str, Any] = {
    "server": {
        "listen_host": "0.0.0.0",
        "listen_port": 8000,
        "rocm_aotriton_experimental": True,
        "require_gpu": True,
        "allow_cpu_fallback": False,
        "preload_default_t2i_on_startup": True,
        "t2v_backend": "auto",
        "t2v_npu_runner": "",
        "t2v_npu_model_dir": "",
    },
    "paths": {
        "models_dir": "models",
        "outputs_dir": "outputs",
        "tmp_dir": "tmp",
        "logs_dir": "logs",
    },
    "huggingface": {
        "token": "",
    },
    "logging": {
        "level": "INFO",
    },
    "defaults": {
        "text2image_model": "runwayml/stable-diffusion-v1-5",
        "image2image_model": "runwayml/stable-diffusion-v1-5",
        "text2video_model": "damo-vilab/text-to-video-ms-1.7b",
        "image2video_model": "ali-vilab/i2vgen-xl",
        "num_inference_steps": 30,
        "num_frames": 16,
        "guidance_scale": 9.0,
        "fps": 8,
        "width": 512,
        "height": 512,
    },
}

TASKS: Dict[str, Dict[str, Any]] = {}
TASKS_LOCK = threading.Lock()
PIPELINES: Dict[str, Any] = {}
PIPELINES_LOCK = threading.Lock()
PIPELINE_LOAD_LOCK = threading.Lock()
VAES: Dict[str, Any] = {}
VAES_LOCK = threading.Lock()
MODEL_SIZE_CACHE: Dict[str, Dict[str, Any]] = {}
MODEL_SIZE_CACHE_LOCK = threading.Lock()
MODEL_SIZE_CACHE_TTL_SEC = 60 * 60
LOGGER = logging.getLogger("videogen")
LOGGER_LOCK = threading.Lock()
LOGGER_READY = False
CURRENT_LOG_FILE: Optional[Path] = None
CURRENT_LOG_LEVEL: Optional[int] = None
PRELOAD_LOCK = threading.Lock()
PRELOAD_STATE: Dict[str, Any] = {
    "running": False,
    "last_trigger": None,
    "last_started_at": None,
    "last_finished_at": None,
    "last_error": None,
    "last_model": None,
}
PROCESS_LOG_TIMESTAMP = time.strftime("%Y%m%d_%H%M%S", time.localtime())
LOG_FILE_NAME = f"{PROCESS_LOG_TIMESTAMP}_videogen_pid{os.getpid()}.log"
LOG_FILE_MAX_BYTES = 10 * 1024 * 1024
LOG_FILE_BACKUP_COUNT = 5
TASK_TRACEBACK_MAX_CHARS = 4000
VALID_LOG_LEVELS = {"DEBUG", "INFO", "WARNING", "ERROR"}
HIGH_VALUE_ENDPOINTS = {
    "/api/generate/text2image",
    "/api/generate/image2image",
    "/api/generate/text2video",
    "/api/generate/image2video",
    "/api/models/loras/catalog",
    "/api/models/download",
    "/api/models/local/delete",
    "/api/outputs/delete",
    "/api/settings",
    "/api/cache/hf/clear",
}
NOISY_DEBUG_ENDPOINT_PREFIXES = ("/api/tasks/",)
TRUE_BOOL_STRINGS = {"1", "true", "yes", "on"}
FALSE_BOOL_STRINGS = {"0", "false", "no", "off"}
SUPPORTED_LOCAL_PIPELINE_CLASSES: Dict[str, set[str]] = {
    "text-to-image": {
        "StableDiffusionPipeline",
        "StableDiffusionXLPipeline",
        "FluxPipeline",
        "PixArtAlphaPipeline",
        "PixArtSigmaPipeline",
        "LatentConsistencyModelPipeline",
        "AuraFlowPipeline",
    },
    "text-to-video": {"TextToVideoSDPipeline"},
    "image-to-image": {
        "StableDiffusionImg2ImgPipeline",
        "StableDiffusionXLImg2ImgPipeline",
        "FluxImg2ImgPipeline",
        "PixArtAlphaPipeline",
        "PixArtSigmaPipeline",
        "LatentConsistencyModelPipeline",
        "AuraFlowPipeline",
        "StableDiffusionPipeline",
        "StableDiffusionXLPipeline",
        "FluxPipeline",
    },
    "image-to-video": {"I2VGenXLPipeline", "WanImageToVideoPipeline"},
}
SUPPORTED_VAE_CLASSES = {"AutoencoderKL", "AsymmetricAutoencoderKL", "AutoencoderTiny", "ConsistencyDecoderVAE"}
LORA_WEIGHT_CANDIDATES = (
    "pytorch_lora_weights.safetensors",
    "pytorch_lora_weights.bin",
)
TASK_DEFAULT_MODEL_KEY: Dict[str, str] = {
    "text-to-image": "text2image_model",
    "image-to-image": "image2image_model",
    "text-to-video": "text2video_model",
    "image-to-video": "image2video_model",
}
CIVITAI_API_BASE = "https://civitai.com/api/v1"
DOWNLOAD_STREAM_CHUNK_BYTES = 1024 * 1024
OUTPUT_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}
OUTPUT_VIDEO_EXTENSIONS = {".mp4", ".mov", ".webm", ".mkv"}
SINGLE_FILE_MODEL_EXTENSIONS = {".safetensors", ".ckpt"}
SUPPORTED_T2V_BACKENDS = {"auto", "cuda", "npu"}


@dataclasses.dataclass
class LocalModelMeta:
    class_name: str
    base_model: Optional[str]
    compatible_tasks: list[str]
    is_lora: bool
    is_vae: bool


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def deep_merge(base: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def parse_bool_setting(raw_value: Any, default: bool = False) -> bool:
    if raw_value is None:
        return default
    if isinstance(raw_value, bool):
        return raw_value
    if isinstance(raw_value, str):
        normalized = raw_value.strip().lower()
        if normalized in TRUE_BOOL_STRINGS:
            return True
        if normalized in FALSE_BOOL_STRINGS:
            return False
        return default
    return bool(raw_value)


def sanitize_settings(payload: Dict[str, Any]) -> Dict[str, Any]:
    cleaned = copy.deepcopy(payload)
    server = cleaned.setdefault("server", {})
    raw_port = server.get("listen_port", 8000)
    try:
        listen_port = int(raw_port)
    except Exception:
        listen_port = 8000
    if listen_port < 1 or listen_port > 65535:
        listen_port = 8000
    server["listen_port"] = listen_port
    listen_host = str(server.get("listen_host", "0.0.0.0")).strip() or "0.0.0.0"
    server["listen_host"] = listen_host
    server["rocm_aotriton_experimental"] = parse_bool_setting(server.get("rocm_aotriton_experimental", True), default=True)
    server["require_gpu"] = parse_bool_setting(server.get("require_gpu", True), default=True)
    server["allow_cpu_fallback"] = parse_bool_setting(server.get("allow_cpu_fallback", False), default=False)
    server["preload_default_t2i_on_startup"] = parse_bool_setting(server.get("preload_default_t2i_on_startup", True), default=True)
    raw_t2v_backend = str(server.get("t2v_backend", "auto")).strip().lower()
    server["t2v_backend"] = raw_t2v_backend if raw_t2v_backend in SUPPORTED_T2V_BACKENDS else "auto"
    server["t2v_npu_runner"] = str(server.get("t2v_npu_runner", "")).strip()
    server["t2v_npu_model_dir"] = str(server.get("t2v_npu_model_dir", "")).strip()
    logging_config = cleaned.setdefault("logging", {})
    raw_level = str(logging_config.get("level", "INFO")).strip().upper()
    logging_config["level"] = raw_level if raw_level in VALID_LOG_LEVELS else "INFO"
    return cleaned


def resolve_path(path_like: str) -> Path:
    candidate = Path(path_like).expanduser()
    if not candidate.is_absolute():
        candidate = BASE_DIR / candidate
    return candidate.resolve()


def get_logs_dir(settings: Dict[str, Any]) -> Path:
    logs_dir_raw = settings.get("paths", {}).get("logs_dir", "logs")
    return resolve_path(str(logs_dir_raw))


def get_log_file_path(settings: Dict[str, Any]) -> Path:
    return get_logs_dir(settings) / LOG_FILE_NAME


def latest_log_file(settings: Dict[str, Any]) -> Optional[Path]:
    logs_dir = get_logs_dir(settings)
    if not logs_dir.exists():
        return None
    candidates = sorted(
        logs_dir.glob("*_videogen_pid*.log"),
        key=lambda p: p.stat().st_mtime if p.exists() else 0,
        reverse=True,
    )
    if candidates:
        return candidates[0]
    fallback = sorted(
        logs_dir.glob("*.log"),
        key=lambda p: p.stat().st_mtime if p.exists() else 0,
        reverse=True,
    )
    return fallback[0] if fallback else None


def setup_logger(settings: Dict[str, Any]) -> None:
    global LOGGER_READY, CURRENT_LOG_FILE, CURRENT_LOG_LEVEL
    log_file = get_log_file_path(settings)
    level_name = str(settings.get("logging", {}).get("level", "INFO")).strip().upper()
    level_value = getattr(logging, level_name, logging.INFO)
    log_file.parent.mkdir(parents=True, exist_ok=True)
    with LOGGER_LOCK:
        if LOGGER_READY and CURRENT_LOG_FILE == log_file and CURRENT_LOG_LEVEL == level_value and LOGGER.handlers:
            return
        for handler in list(LOGGER.handlers):
            LOGGER.removeHandler(handler)
            handler.close()
        LOGGER.setLevel(level_value)
        LOGGER.propagate = False
        formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(threadName)s | %(message)s")
        file_handler = RotatingFileHandler(
            filename=str(log_file),
            maxBytes=LOG_FILE_MAX_BYTES,
            backupCount=LOG_FILE_BACKUP_COUNT,
            encoding="utf-8",
        )
        file_handler.setLevel(level_value)
        file_handler.setFormatter(formatter)
        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(level_value)
        stream_handler.setFormatter(formatter)
        LOGGER.addHandler(file_handler)
        LOGGER.addHandler(stream_handler)
        CURRENT_LOG_FILE = log_file
        CURRENT_LOG_LEVEL = level_value
        LOGGER_READY = True
    LOGGER.info("logger initialized file=%s level=%s", str(log_file), level_name)


def format_exception_trace(limit_chars: int = TASK_TRACEBACK_MAX_CHARS) -> str:
    trace = traceback.format_exc()
    if len(trace) <= limit_chars:
        return trace
    return trace[:limit_chars] + "\n... (trace truncated) ..."


def runtime_diagnostics() -> Dict[str, Any]:
    info = detect_runtime()
    details: Dict[str, Any] = {
        "device": info.get("device"),
        "cuda_available": info.get("cuda_available"),
        "rocm_available": info.get("rocm_available"),
        "torch_version": info.get("torch_version"),
    }
    if info.get("import_error"):
        details["import_error"] = info.get("import_error")
        return details
    try:
        if torch.cuda.is_available():
            free_bytes, total_bytes = torch.cuda.mem_get_info()
            details["gpu_free_bytes"] = int(free_bytes)
            details["gpu_total_bytes"] = int(total_bytes)
    except Exception:
        pass
    return details


def gpu_memory_stats() -> Dict[str, int]:
    stats: Dict[str, int] = {}
    if TORCH_IMPORT_ERROR or not torch.cuda.is_available():
        return stats
    try:
        free_bytes, total_bytes = torch.cuda.mem_get_info()
        stats["gpu_free_bytes"] = int(free_bytes)
        stats["gpu_total_bytes"] = int(total_bytes)
    except Exception:
        pass
    try:
        stats["torch_allocated_bytes"] = int(torch.cuda.memory_allocated())
    except Exception:
        pass
    try:
        stats["torch_reserved_bytes"] = int(torch.cuda.memory_reserved())
    except Exception:
        pass
    return stats


def log_gpu_memory_stats(label: str, task_id: str) -> None:
    stats = gpu_memory_stats()
    if not stats:
        return
    LOGGER.debug(
        "gpu memory task_id=%s label=%s free=%s total=%s allocated=%s reserved=%s",
        task_id,
        label,
        stats.get("gpu_free_bytes"),
        stats.get("gpu_total_bytes"),
        stats.get("torch_allocated_bytes"),
        stats.get("torch_reserved_bytes"),
    )


def normalize_user_path(path_like: str) -> Path:
    return Path(path_like).expanduser().resolve()


def gather_hf_cache_candidates() -> set[Path]:
    candidates: set[Path] = set()
    hf_home_env = os.environ.get("HF_HOME", "").strip()
    if hf_home_env:
        hf_home = normalize_user_path(hf_home_env)
    else:
        hf_home = normalize_user_path(str(Path.home() / ".cache" / "huggingface"))

    hub_cache_env = os.environ.get("HUGGINGFACE_HUB_CACHE", "").strip()
    if hub_cache_env:
        candidates.add(normalize_user_path(hub_cache_env))
    else:
        candidates.add(hf_home / "hub")

    transformers_cache_env = os.environ.get("TRANSFORMERS_CACHE", "").strip()
    if transformers_cache_env:
        candidates.add(normalize_user_path(transformers_cache_env))
    else:
        candidates.add(hf_home / "transformers")

    for name in ("assets", "xet", "modules"):
        candidates.add(hf_home / name)

    try:
        from huggingface_hub import constants as hf_constants

        for attr in ("HUGGINGFACE_HUB_CACHE", "HF_ASSETS_CACHE"):
            value = getattr(hf_constants, attr, None)
            if isinstance(value, str) and value.strip():
                candidates.add(normalize_user_path(value))
    except Exception:
        pass

    try:
        from transformers.utils import TRANSFORMERS_CACHE

        if isinstance(TRANSFORMERS_CACHE, str) and TRANSFORMERS_CACHE.strip():
            candidates.add(normalize_user_path(TRANSFORMERS_CACHE))
    except Exception:
        pass

    return {path for path in candidates if str(path).strip()}


def is_safe_cache_target(path: Path) -> bool:
    if not path.is_absolute():
        return False
    if path.parent == path:
        return False
    anchor = Path(path.anchor)
    if path == anchor:
        return False
    home = Path.home().resolve()
    if path == home:
        return False
    return True


def is_deletable_model_dir(model_dir: Path) -> bool:
    if not model_dir.exists() or not model_dir.is_dir():
        return False
    if (model_dir / ".cache" / "huggingface").exists():
        return True
    if (model_dir / "model_index.json").exists():
        return True
    if any(model_dir.glob("*.safetensors")):
        return True
    return False


def sanitize_repo_id(repo_id: str) -> str:
    return repo_id.replace("/", "--")


def desanitize_repo_id(name: str) -> str:
    return name.replace("--", "/")


def safe_in_directory(target: Path, root: Path) -> bool:
    try:
        target.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def detect_output_kind(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in OUTPUT_IMAGE_EXTENSIONS:
        return "image"
    if suffix in OUTPUT_VIDEO_EXTENSIONS:
        return "video"
    return "other"


def output_view_url(file_name: str, kind: str) -> Optional[str]:
    if kind == "image":
        return f"/api/images/{quote(file_name, safe='')}"
    if kind == "video":
        return f"/api/videos/{quote(file_name, safe='')}"
    return None


class SettingsStore:
    def __init__(self, path: Path, defaults: Dict[str, Any]) -> None:
        self._path = path
        self._defaults = copy.deepcopy(defaults)
        self._lock = threading.Lock()
        self._path.parent.mkdir(parents=True, exist_ok=True)
        loaded, should_persist = self._load()
        self._settings = loaded
        if should_persist:
            self._write(self._settings)

    def _load(self) -> tuple[Dict[str, Any], bool]:
        if not self._path.exists():
            return sanitize_settings(copy.deepcopy(self._defaults)), True
        try:
            content = json.loads(self._path.read_text(encoding="utf-8"))
            if not isinstance(content, dict):
                return sanitize_settings(copy.deepcopy(self._defaults)), True
            merged = sanitize_settings(deep_merge(self._defaults, content))
            should_persist = merged != content
            return merged, should_persist
        except Exception:
            return sanitize_settings(copy.deepcopy(self._defaults)), True

    def _write(self, payload: Dict[str, Any]) -> None:
        tmp = self._path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        tmp.replace(self._path)

    def get(self) -> Dict[str, Any]:
        with self._lock:
            return copy.deepcopy(self._settings)

    def update(self, updates: Dict[str, Any]) -> Dict[str, Any]:
        with self._lock:
            merged = deep_merge(self._settings, updates)
            self._settings = sanitize_settings(deep_merge(self._defaults, merged))
            self._write(self._settings)
            return copy.deepcopy(self._settings)


settings_store = SettingsStore(DATA_DIR / "settings.json", DEFAULT_SETTINGS)


def ensure_runtime_dirs(settings: Dict[str, Any]) -> None:
    for key in ("models_dir", "outputs_dir", "tmp_dir", "logs_dir"):
        resolve_path(settings["paths"][key]).mkdir(parents=True, exist_ok=True)


ensure_runtime_dirs(settings_store.get())
setup_logger(settings_store.get())


def server_flag(name: str, default: bool) -> bool:
    try:
        settings = settings_store.get()
    except Exception:
        return default
    return parse_bool_setting(settings.get("server", {}).get(name, default), default=default)


def detect_runtime() -> Dict[str, Any]:
    settings = settings_store.get()
    configured_t2v_backend = str(settings.get("server", {}).get("t2v_backend", "auto")).strip().lower()
    if configured_t2v_backend not in SUPPORTED_T2V_BACKENDS:
        configured_t2v_backend = "auto"
    npu_runner_configured = bool(str(settings.get("server", {}).get("t2v_npu_runner", "")).strip())
    npu_available = False
    npu_backend = ""
    npu_reason = "onnxruntime/ryzenai runtime is not installed"
    if importlib.util.find_spec("onnxruntime") is not None:
        try:
            import onnxruntime as ort  # type: ignore

            providers = [str(p) for p in ort.get_available_providers()]
            npu_provider = next(
                (
                    p
                    for p in providers
                    if ("NPU" in p.upper()) or ("VITISAI" in p.upper()) or ("RYZENAI" in p.upper())
                ),
                "",
            )
            if npu_provider:
                npu_available = True
                npu_backend = npu_provider
                npu_reason = ""
            else:
                npu_reason = f"onnxruntime providers={providers}"
        except Exception as exc:  # pragma: no cover
            npu_reason = f"onnxruntime probe failed: {exc}"

    if TORCH_IMPORT_ERROR:
        return {
            "diffusers_ready": False,
            "cuda_available": False,
            "rocm_available": False,
            "npu_available": npu_available,
            "npu_backend": npu_backend,
            "npu_reason": npu_reason,
            "t2v_backend_default": configured_t2v_backend,
            "t2v_npu_runner_configured": npu_runner_configured,
            "device": "cpu",
            "import_error": TORCH_IMPORT_ERROR,
        }
    cuda_available = bool(torch.cuda.is_available())
    rocm_available = bool(getattr(torch.version, "hip", None))
    diffusers_error = DIFFUSERS_IMPORT_ERROR
    if diffusers_error is None:
        try:
            load_diffusers_components()
        except Exception as exc:  # pragma: no cover
            diffusers_error = str(exc)
    return {
        "diffusers_ready": diffusers_error is None,
        "cuda_available": cuda_available,
        "rocm_available": rocm_available,
        "npu_available": npu_available,
        "npu_backend": npu_backend,
        "npu_reason": npu_reason,
        "t2v_backend_default": configured_t2v_backend,
        "t2v_npu_runner_configured": npu_runner_configured,
        "device": "cuda" if cuda_available else "cpu",
        "torch_version": torch.__version__,
        "rocm_aotriton_env": os.environ.get("TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL", ""),
        "require_gpu": server_flag("require_gpu", True),
        "allow_cpu_fallback": server_flag("allow_cpu_fallback", False),
        "import_error": diffusers_error,
    }


def resolve_text2video_backend(requested_backend: str, settings: Dict[str, Any]) -> Literal["cuda", "npu"]:
    requested = str(requested_backend or "").strip().lower()
    if requested not in SUPPORTED_T2V_BACKENDS:
        requested = "auto"
    configured = str(settings.get("server", {}).get("t2v_backend", "auto")).strip().lower()
    if configured not in SUPPORTED_T2V_BACKENDS:
        configured = "auto"
    selected = requested if requested != "auto" else configured
    if selected == "auto":
        runtime = detect_runtime()
        has_runner = bool(str(settings.get("server", {}).get("t2v_npu_runner", "")).strip())
        if bool(runtime.get("npu_available")) and has_runner:
            return "npu"
        return "cuda"
    if selected == "npu":
        return "npu"
    return "cuda"


def run_text2video_npu_runner(task_id: str, payload: "Text2VideoRequest", settings: Dict[str, Any], model_ref: str) -> Dict[str, Any]:
    runner_raw = str(settings.get("server", {}).get("t2v_npu_runner", "")).strip()
    if not runner_raw:
        raise RuntimeError(
            "NPU backend requires server.t2v_npu_runner. "
            "Configure a runner executable/script path in Settings."
        )
    runner_path = Path(runner_raw).expanduser()
    if not runner_path.is_absolute():
        runner_path = (BASE_DIR / runner_path).resolve()
    if not runner_path.exists():
        raise RuntimeError(f"NPU runner not found: {runner_path}")

    npu_model_dir_raw = str(settings.get("server", {}).get("t2v_npu_model_dir", "")).strip()
    npu_model_dir = ""
    if npu_model_dir_raw:
        npu_model_dir = str(resolve_path(npu_model_dir_raw))
    else:
        resolved_model = Path(resolve_model_source(model_ref, settings))
        if resolved_model.exists() and resolved_model.is_dir():
            npu_model_dir = str(resolved_model)

    output_name = f"text2video_{task_id}.mp4"
    output_path = resolve_path(settings["paths"]["outputs_dir"]) / output_name
    tmp_dir = resolve_path(settings["paths"]["tmp_dir"])
    tmp_dir.mkdir(parents=True, exist_ok=True)
    req_path = tmp_dir / f"{task_id}_npu_t2v_request.json"
    req_payload = {
        "task_id": task_id,
        "prompt": payload.prompt,
        "negative_prompt": payload.negative_prompt or "",
        "model_id": model_ref,
        "npu_model_dir": npu_model_dir,
        "num_inference_steps": int(payload.num_inference_steps),
        "num_frames": int(payload.num_frames),
        "guidance_scale": float(payload.guidance_scale),
        "fps": int(payload.fps),
        "seed": payload.seed,
        "output_path": str(output_path),
    }
    req_path.write_text(json.dumps(req_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    if runner_path.suffix.lower() == ".py":
        cmd = [sys.executable, str(runner_path), "--input-json", str(req_path)]
    elif runner_path.suffix.lower() in (".bat", ".cmd"):
        cmd = ["cmd.exe", "/c", str(runner_path), "--input-json", str(req_path)]
    else:
        cmd = [str(runner_path), "--input-json", str(req_path)]

    update_task(task_id, progress=0.35, message="Generating frames (NPU runner)")
    runtime = detect_runtime()
    if not bool(runtime.get("npu_available")):
        LOGGER.warning(
            "local npu runtime probe is unavailable; continuing via external runner task_id=%s reason=%s",
            task_id,
            runtime.get("npu_reason"),
        )
    LOGGER.info(
        "text2video npu runner start task_id=%s backend=npu runner=%s npu_model_dir=%s output=%s",
        task_id,
        str(runner_path),
        npu_model_dir or "(none)",
        str(output_path),
    )
    started = time.perf_counter()
    completed = subprocess.run(
        cmd,
        cwd=str(BASE_DIR),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=60 * 60 * 4,
        check=False,
    )
    elapsed_ms = (time.perf_counter() - started) * 1000
    stdout_tail = (completed.stdout or "")[-4000:]
    stderr_tail = (completed.stderr or "")[-4000:]
    LOGGER.info(
        "text2video npu runner done task_id=%s return_code=%s elapsed_ms=%.1f",
        task_id,
        completed.returncode,
        elapsed_ms,
    )
    if completed.stdout:
        LOGGER.debug("text2video npu runner stdout task_id=%s tail=%s", task_id, stdout_tail)
    if completed.stderr:
        LOGGER.debug("text2video npu runner stderr task_id=%s tail=%s", task_id, stderr_tail)
    if completed.returncode != 0:
        raise RuntimeError(
            f"NPU runner failed (exit={completed.returncode}). "
            f"stderr_tail={stderr_tail or '(none)'}"
        )
    if not output_path.exists():
        raise RuntimeError(f"NPU runner completed but output video not found: {output_path}")
    return {"video_file": output_name, "encoder": "npu_runner", "runner": str(runner_path)}


def get_device_and_dtype() -> tuple[str, Any]:
    if TORCH_IMPORT_ERROR:
        raise RuntimeError(f"Diffusers runtime is not available: {TORCH_IMPORT_ERROR}")
    cuda_available = bool(torch.cuda.is_available())
    if not cuda_available:
        raise RuntimeError(
            "GPU is unavailable (torch.cuda.is_available() is false). "
            "CPU fallback is disabled for root-cause diagnostics."
        )
    return "cuda", torch.float16


def assert_generation_runtime_ready() -> None:
    if TORCH_IMPORT_ERROR:
        raise RuntimeError(f"Runtime is not available: {TORCH_IMPORT_ERROR}")
    load_diffusers_components()
    # Fail fast when GPU is required but unavailable.
    get_device_and_dtype()


def load_diffusers_components() -> Dict[str, Any]:
    global DIFFUSERS_IMPORT_ERROR
    if DIFFUSERS_COMPONENTS:
        return DIFFUSERS_COMPONENTS
    if TORCH_IMPORT_ERROR:
        raise RuntimeError(f"Runtime is not available: {TORCH_IMPORT_ERROR}")
    try:
        from diffusers import (
            AutoPipelineForImage2Image,
            AutoPipelineForText2Image,
            AutoencoderKL,
            DPMSolverMultistepScheduler,
            I2VGenXLPipeline,
            StableDiffusionImg2ImgPipeline,
            StableDiffusionPipeline,
            StableDiffusionXLImg2ImgPipeline,
            StableDiffusionXLPipeline,
            TextToVideoSDPipeline,
            WanImageToVideoPipeline,
        )
        from diffusers.utils import export_to_video
    except Exception as exc:
        DIFFUSERS_IMPORT_ERROR = str(exc)
        raise RuntimeError(f"Diffusers import failed: {exc}") from exc

    DIFFUSERS_COMPONENTS.update(
        {
            "AutoPipelineForImage2Image": AutoPipelineForImage2Image,
            "AutoPipelineForText2Image": AutoPipelineForText2Image,
            "AutoencoderKL": AutoencoderKL,
            "DPMSolverMultistepScheduler": DPMSolverMultistepScheduler,
            "I2VGenXLPipeline": I2VGenXLPipeline,
            "StableDiffusionImg2ImgPipeline": StableDiffusionImg2ImgPipeline,
            "StableDiffusionPipeline": StableDiffusionPipeline,
            "StableDiffusionXLImg2ImgPipeline": StableDiffusionXLImg2ImgPipeline,
            "StableDiffusionXLPipeline": StableDiffusionXLPipeline,
            "TextToVideoSDPipeline": TextToVideoSDPipeline,
            "WanImageToVideoPipeline": WanImageToVideoPipeline,
            "export_to_video": export_to_video,
        }
    )
    DIFFUSERS_IMPORT_ERROR = None
    return DIFFUSERS_COMPONENTS


def create_task(task_type: str, message: str = "Queued") -> str:
    task_id = str(uuid.uuid4())
    now = utc_now()
    with TASKS_LOCK:
        TASKS[task_id] = {
            "id": task_id,
            "task_type": task_type,
            "status": "queued",
            "progress": 0.0,
            "message": message,
            "created_at": now,
            "updated_at": now,
            "started_at": None,
            "finished_at": None,
            "result": None,
            "error": None,
            "downloaded_bytes": None,
            "total_bytes": None,
        }
    LOGGER.info("task created id=%s type=%s message=%s", task_id, task_type, message)
    return task_id


def update_task(task_id: str, **updates: Any) -> None:
    with TASKS_LOCK:
        if task_id not in TASKS:
            return
        previous = copy.deepcopy(TASKS[task_id])
        TASKS[task_id].update(updates)
        status = str(TASKS[task_id].get("status") or "")
        if status == "running" and not TASKS[task_id].get("started_at"):
            TASKS[task_id]["started_at"] = utc_now()
        if status in ("completed", "error"):
            TASKS[task_id]["finished_at"] = TASKS[task_id].get("finished_at") or utc_now()
        TASKS[task_id]["updated_at"] = utc_now()
        current = copy.deepcopy(TASKS[task_id])
    if previous.get("status") != current.get("status"):
        LOGGER.info(
            "task status id=%s type=%s %s->%s progress=%.3f message=%s",
            task_id,
            current.get("task_type"),
            previous.get("status"),
            current.get("status"),
            float(current.get("progress") or 0.0),
            current.get("message"),
        )
    if updates.get("error"):
        LOGGER.error("task error id=%s error=%s", task_id, updates.get("error"))


@contextlib.contextmanager
def task_progress_heartbeat(
    task_id: str,
    start_progress: float,
    end_progress: float,
    message: str,
    *,
    interval_sec: float = 0.5,
    estimated_duration_sec: float = 20.0,
) -> Any:
    """
    Keep task progress moving during long, non-step phases (e.g. decode/encode)
    so the UI does not look frozen around 90%.
    """
    start = max(0.0, min(1.0, float(start_progress)))
    end = max(start, min(1.0, float(end_progress)))
    span = max(0.0, end - start)
    if span <= 0.0:
        yield
        return

    stop_event = threading.Event()
    state = {"last_progress": start}

    def worker() -> None:
        started_at = time.perf_counter()
        expected = max(0.5, float(estimated_duration_sec))
        tick = 0
        while not stop_event.wait(max(0.2, float(interval_sec))):
            elapsed = time.perf_counter() - started_at
            ratio = min(elapsed / expected, 0.985)
            progress = start + span * ratio
            if progress <= state["last_progress"] + 1e-6:
                continue
            state["last_progress"] = progress
            update_task(task_id, progress=progress, message=message)
            tick += 1
            if tick % 10 == 0:
                LOGGER.debug(
                    "task heartbeat task_id=%s message=%s progress=%.4f elapsed_sec=%.1f",
                    task_id,
                    message,
                    progress,
                    elapsed,
                )

    heartbeat = threading.Thread(
        target=worker,
        name=f"task-hb-{task_id[:8]}",
        daemon=True,
    )
    heartbeat.start()
    try:
        yield
    finally:
        stop_event.set()
        heartbeat.join(timeout=2.0)


def get_task(task_id: str) -> Optional[Dict[str, Any]]:
    with TASKS_LOCK:
        task = TASKS.get(task_id)
        return copy.deepcopy(task) if task else None


def resolve_model_source(model_ref: str, settings: Dict[str, Any]) -> str:
    direct = Path(model_ref).expanduser()
    if direct.exists():
        return str(direct.resolve())
    models_dir = resolve_path(settings["paths"]["models_dir"])
    local_dir = models_dir / sanitize_repo_id(model_ref)
    if local_dir.exists():
        return str(local_dir.resolve())
    return model_ref


def resolve_lora_source(lora_ref: str, settings: Dict[str, Any]) -> str:
    direct = Path(lora_ref).expanduser()
    if direct.exists():
        return str(direct.resolve())
    models_dir = resolve_path(settings["paths"]["models_dir"])
    local_dir = models_dir / sanitize_repo_id(lora_ref)
    if local_dir.exists():
        return str(local_dir.resolve())
    return lora_ref


def is_single_file_model(path: Path) -> bool:
    return path.exists() and path.is_file() and path.suffix.lower() in SINGLE_FILE_MODEL_EXTENSIONS


def infer_single_file_family(path: Path) -> str:
    name = path.name.lower()
    if "sdxl" in name or "xl" in name:
        return "sdxl"
    return "sd"


def find_local_sdxl_base_config(settings: Dict[str, Any]) -> Optional[str]:
    models_dir = resolve_path(settings["paths"]["models_dir"])
    candidates = (
        models_dir / "stabilityai--stable-diffusion-xl-base-1.0",
        models_dir / "stabilityai--stable-diffusion-xl-base-1-0",
    )
    for candidate in candidates:
        if candidate.exists() and candidate.is_dir():
            return str(candidate.resolve())
    return None


def single_file_model_compatible_tasks(path: Path) -> list[str]:
    if not is_single_file_model(path):
        return []
    family = infer_single_file_family(path)
    if family in ("sd", "sdxl"):
        return ["text-to-image", "image-to-image"]
    return []


def single_file_base_model_label(path: Path) -> Optional[str]:
    if not is_single_file_model(path):
        return None
    family = infer_single_file_family(path)
    if family == "sdxl":
        return "SDXL"
    if family == "sd":
        return "SD 1.x/2.x"
    return None


def parse_lora_adapter_config(lora_dir: Path) -> Dict[str, Any]:
    config_path = lora_dir / "adapter_config.json"
    if not config_path.exists() or not config_path.is_file():
        return {}
    try:
        payload = json.loads(config_path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def is_local_lora_dir(path: Path) -> bool:
    if not path.is_dir():
        return False
    model_index = load_local_model_index(path)
    class_name = str((model_index or {}).get("_class_name") or "").strip()
    if class_name:
        # If this is already a known base pipeline class, do not treat it as LoRA
        # even when sample files include "lora" in their names.
        for allowed in SUPPORTED_LOCAL_PIPELINE_CLASSES.values():
            if class_name in allowed:
                return False
    if (path / "adapter_config.json").exists():
        return True
    for name in LORA_WEIGHT_CANDIDATES:
        if (path / name).exists():
            return True
    for candidate in path.glob("*.safetensors"):
        if "lora" in candidate.name.lower():
            return True
    return False


def normalize_ref(value: str) -> str:
    return value.strip().replace("\\", "/").rstrip("/").lower()


def model_ref_candidates(model_ref: str) -> set[str]:
    candidates: set[str] = set()
    raw = (model_ref or "").strip()
    if not raw:
        return candidates
    candidates.add(normalize_ref(raw))
    candidates.add(normalize_ref(desanitize_repo_id(Path(raw).name)))
    direct = Path(raw).expanduser()
    if direct.exists():
        resolved = direct.resolve()
        candidates.add(normalize_ref(str(resolved)))
        candidates.add(normalize_ref(desanitize_repo_id(resolved.name)))
    return candidates


def lora_matches_model(base_model_hint: str, model_ref: str) -> bool:
    hint = normalize_ref(base_model_hint or "")
    if not hint:
        return True
    if not model_ref.strip():
        return True
    candidates = model_ref_candidates(model_ref)
    if hint in candidates:
        return True
    hint_tail = normalize_ref(Path(hint).name)
    if hint_tail and hint_tail in candidates:
        return True
    for item in candidates:
        if item.endswith(hint_tail) and hint_tail:
            return True
    return False


def is_lora_error_compatible_skip(exc: Exception) -> bool:
    text = str(exc or "").lower()
    if "load_lora_adapter" in text:
        return True
    if "unet3dconditionmodel" in text:
        return True
    if "does not support lora" in text:
        return True
    return False


def local_lora_size_bytes(lora_dir: Path) -> Optional[int]:
    total = 0
    found = False
    for path in lora_dir.rglob("*"):
        if path.is_file():
            found = True
            try:
                total += path.stat().st_size
            except OSError:
                pass
    return total if found else None


def collect_lora_refs(lora_id: Optional[str], lora_ids: Optional[list[str]]) -> list[str]:
    refs: list[str] = []
    for raw in ([lora_id] if lora_id else []) + list(lora_ids or []):
        value = str(raw or "").strip()
        if not value:
            continue
        if value not in refs:
            refs.append(value)
    return refs


def apply_loras_to_pipeline(pipe: Any, lora_refs: list[str], lora_scale: float, settings: Dict[str, Any]) -> list[str]:
    if not lora_refs:
        # Avoid calling unload on every request. Some pipelines implement
        # unload paths incompletely and this can create expensive warnings.
        if getattr(pipe, "_videogen_lora_loaded", False) and hasattr(pipe, "unload_lora_weights"):
            try:
                pipe.unload_lora_weights()
            except Exception:
                LOGGER.warning("unload_lora_weights failed; continuing", exc_info=True)
            finally:
                pipe._videogen_lora_loaded = False
        return []
    if not hasattr(pipe, "load_lora_weights"):
        raise RuntimeError("Selected model does not support LoRA")
    if getattr(pipe, "_videogen_lora_loaded", False) and hasattr(pipe, "unload_lora_weights"):
        try:
            pipe.unload_lora_weights()
        except Exception:
            LOGGER.warning("unload_lora_weights failed; continuing", exc_info=True)
        finally:
            pipe._videogen_lora_loaded = False

    loaded_sources: list[str] = []
    adapter_names: list[str] = []
    for idx, chosen in enumerate(lora_refs):
        adapter_name = f"selected_lora_{idx}"
        source = resolve_lora_source(chosen, settings)
        source_path = Path(source)
        if source_path.exists() and source_path.is_file():
            pipe.load_lora_weights(str(source_path.parent), weight_name=source_path.name, adapter_name=adapter_name)
        else:
            pipe.load_lora_weights(source, adapter_name=adapter_name)
        loaded_sources.append(source)
        adapter_names.append(adapter_name)
    if hasattr(pipe, "set_adapters"):
        try:
            pipe.set_adapters(adapter_names, adapter_weights=[float(lora_scale)] * len(adapter_names))
        except Exception:
            LOGGER.warning("set_adapters failed; continuing with default LoRA weight", exc_info=True)
    elif len(adapter_names) > 1:
        LOGGER.warning("multiple LoRAs selected but pipeline has no set_adapters; behavior depends on pipeline implementation")
    pipe._videogen_lora_loaded = bool(adapter_names)
    return loaded_sources


def is_local_vae_dir(path: Path) -> bool:
    model_index = load_local_model_index(path)
    if not model_index:
        return False
    class_name = str(model_index.get("_class_name") or "").strip()
    return class_name in SUPPORTED_VAE_CLASSES


def resolve_vae_source(vae_ref: str, settings: Dict[str, Any]) -> str:
    return resolve_model_source(vae_ref, settings)


def get_vae(vae_ref: str, settings: Dict[str, Any], device: str, dtype: Any) -> Any:
    source = resolve_vae_source(vae_ref, settings)
    with VAES_LOCK:
        cached = VAES.get(source)
        if cached is not None:
            return cached
    components = load_diffusers_components()
    AutoencoderKL = components["AutoencoderKL"]
    vae = AutoencoderKL.from_pretrained(source, torch_dtype=dtype)
    vae = vae.to(device)
    with VAES_LOCK:
        VAES[source] = vae
    return vae


def apply_vae_to_pipeline(pipe: Any, vae_ref: str, settings: Dict[str, Any], device: str, dtype: Any) -> Optional[str]:
    selected = (vae_ref or "").strip()
    if not hasattr(pipe, "vae"):
        return None
    if not selected:
        default_vae = getattr(pipe, "_videogen_default_vae", None)
        if default_vae is not None:
            pipe.vae = default_vae
        return None
    vae = get_vae(selected, settings, device=device, dtype=dtype)
    if getattr(pipe, "_videogen_default_vae", None) is None:
        pipe._videogen_default_vae = pipe.vae
    pipe.vae = vae
    return resolve_vae_source(selected, settings)


def get_default_model_for_task(task: str, settings: Dict[str, Any]) -> str:
    key = TASK_DEFAULT_MODEL_KEY.get(task, "")
    if not key:
        return ""
    return str(settings.get("defaults", {}).get(key, ""))


def decode_latents_to_pil_images(pipe: Any, latents: Any) -> list[Any]:
    if not hasattr(pipe, "vae") or not hasattr(pipe.vae, "decode"):
        raise RuntimeError("Pipeline VAE decode is not available")
    vae = pipe.vae
    vae_param = next(vae.parameters(), None)
    original_device = vae_param.device if vae_param is not None else torch.device("cpu")
    original_dtype = vae_param.dtype if vae_param is not None else torch.float32
    decode_device = original_device
    decode_dtype = original_dtype
    scaled = latents
    if getattr(scaled, "device", None) != decode_device or getattr(scaled, "dtype", None) != decode_dtype:
        scaled = scaled.to(device=decode_device, dtype=decode_dtype)
    scaling_factor = float(getattr(getattr(vae, "config", None), "scaling_factor", 1.0) or 1.0)
    if scaling_factor not in (0.0, 1.0):
        scaled = scaled / scaling_factor
    with torch.inference_mode():
        decoded = vae.decode(scaled, return_dict=False)[0]
    if hasattr(pipe, "image_processor") and hasattr(pipe.image_processor, "postprocess"):
        images = pipe.image_processor.postprocess(decoded, output_type="pil")
        if isinstance(images, list):
            return images
    if hasattr(pipe, "numpy_to_pil"):
        array = (decoded / 2 + 0.5).clamp(0, 1).detach().float().cpu().permute(0, 2, 3, 1).numpy()
        return pipe.numpy_to_pil(array)
    raise RuntimeError("Pipeline image postprocess is not available")


def apply_rocm_attention_override(pipe: Any) -> None:
    if TORCH_IMPORT_ERROR or not bool(getattr(torch.version, "hip", None)):
        return
    try:
        from diffusers.models.attention_processor import AttnProcessor
    except Exception:
        LOGGER.debug("rocm attention override unavailable", exc_info=True)
        return
    applied: list[str] = []
    for name in ("unet", "transformer"):
        component = getattr(pipe, name, None)
        if component is None or not hasattr(component, "set_attn_processor"):
            continue
        try:
            component.set_attn_processor(AttnProcessor())
            applied.append(name)
        except Exception:
            LOGGER.debug("rocm attention override failed component=%s", name, exc_info=True)
    if applied:
        LOGGER.info("rocm attention override applied components=%s processor=AttnProcessor", ",".join(applied))


def should_apply_rocm_attention_override() -> bool:
    # Default is disabled because this path can degrade throughput severely on some ROCm stacks.
    return parse_bool_setting(os.environ.get("VIDEOGEN_ROCM_ATTN_OVERRIDE", "0"), default=False)


def get_pipeline(kind: Literal["text-to-image", "image-to-image", "text-to-video", "image-to-video"], model_ref: str, settings: Dict[str, Any]) -> Any:
    source = resolve_model_source(model_ref, settings)
    source_path = Path(source)
    source_is_single_file = is_single_file_model(source_path)
    cache_key = f"{kind}:{source}"
    with PIPELINES_LOCK:
        if cache_key in PIPELINES:
            LOGGER.info("pipeline cache hit kind=%s source=%s", kind, source)
            return PIPELINES[cache_key]
    device, dtype = get_device_and_dtype()
    components = load_diffusers_components()
    AutoPipelineForText2Image = components["AutoPipelineForText2Image"]
    AutoPipelineForImage2Image = components["AutoPipelineForImage2Image"]
    StableDiffusionPipeline = components["StableDiffusionPipeline"]
    StableDiffusionImg2ImgPipeline = components["StableDiffusionImg2ImgPipeline"]
    StableDiffusionXLPipeline = components["StableDiffusionXLPipeline"]
    StableDiffusionXLImg2ImgPipeline = components["StableDiffusionXLImg2ImgPipeline"]
    TextToVideoSDPipeline = components["TextToVideoSDPipeline"]
    I2VGenXLPipeline = components["I2VGenXLPipeline"]
    WanImageToVideoPipeline = components["WanImageToVideoPipeline"]
    DPMSolverMultistepScheduler = components["DPMSolverMultistepScheduler"]
    load_started = time.perf_counter()
    LOGGER.info("pipeline load start kind=%s model_ref=%s source=%s device=%s dtype=%s", kind, model_ref, source, device, dtype)
    with PIPELINE_LOAD_LOCK:
        with PIPELINES_LOCK:
            existing = PIPELINES.get(cache_key)
            if existing is not None:
                LOGGER.info("pipeline cache hit(after-wait) kind=%s source=%s", kind, source)
                return existing
        if kind == "text-to-image":
            if source_is_single_file:
                family = infer_single_file_family(source_path)
                if family == "sdxl":
                    config = find_local_sdxl_base_config(settings)
                    kwargs: Dict[str, Any] = {"torch_dtype": dtype}
                    if config:
                        kwargs["config"] = config
                    pipe = StableDiffusionXLPipeline.from_single_file(source, **kwargs)
                else:
                    pipe = StableDiffusionPipeline.from_single_file(source, torch_dtype=dtype)
            else:
                pipe = AutoPipelineForText2Image.from_pretrained(source, torch_dtype=dtype)
        elif kind == "image-to-image":
            if source_is_single_file:
                family = infer_single_file_family(source_path)
                if family == "sdxl":
                    config = find_local_sdxl_base_config(settings)
                    kwargs = {"torch_dtype": dtype}
                    if config:
                        kwargs["config"] = config
                    pipe = StableDiffusionXLImg2ImgPipeline.from_single_file(source, **kwargs)
                else:
                    pipe = StableDiffusionImg2ImgPipeline.from_single_file(source, torch_dtype=dtype)
            else:
                pipe = AutoPipelineForImage2Image.from_pretrained(source, torch_dtype=dtype)
        elif kind == "text-to-video":
            pipe = TextToVideoSDPipeline.from_pretrained(source, torch_dtype=dtype)
            if hasattr(pipe, "scheduler") and hasattr(pipe.scheduler, "config"):
                pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
        else:
            class_name = ""
            if source_path.exists() and source_path.is_dir():
                model_index = load_local_model_index(source_path)
                class_name = str((model_index or {}).get("_class_name") or "").strip()
            if class_name == "WanImageToVideoPipeline":
                pipe = WanImageToVideoPipeline.from_pretrained(source, torch_dtype=dtype)
            else:
                pipe = I2VGenXLPipeline.from_pretrained(source, torch_dtype=dtype)
        pipe = pipe.to(device)
        if hasattr(pipe, "set_progress_bar_config"):
            with contextlib.suppress(Exception):
                pipe.set_progress_bar_config(disable=True)
        if device == "cuda" and bool(getattr(torch.version, "hip", None)) and should_apply_rocm_attention_override():
            apply_rocm_attention_override(pipe)
        if hasattr(pipe, "vae") and getattr(pipe, "_videogen_default_vae", None) is None:
            pipe._videogen_default_vae = pipe.vae
        # On GPU, prefer full attention path to keep utilization on accelerator.
        if device != "cuda" and hasattr(pipe, "enable_attention_slicing"):
            pipe.enable_attention_slicing()
        # channels_last is beneficial on many NVIDIA setups, but on ROCm/AMD it can
        # massively degrade throughput for SDXL-class models.
        if device == "cuda" and not bool(getattr(torch.version, "hip", None)) and hasattr(pipe, "unet"):
            try:
                pipe.unet.to(memory_format=torch.channels_last)
                #pipe.unet.to(memory_format=torch.contiguous_format)   
            except Exception:
                LOGGER.debug("channels_last optimization skipped", exc_info=True)
        with PIPELINES_LOCK:
            PIPELINES[cache_key] = pipe
    LOGGER.info("pipeline load done kind=%s source=%s elapsed_ms=%.1f", kind, source, (time.perf_counter() - load_started) * 1000)
    return pipe


def get_preload_state() -> Dict[str, Any]:
    with PRELOAD_LOCK:
        return copy.deepcopy(PRELOAD_STATE)


def _set_preload_state(**updates: Any) -> None:
    with PRELOAD_LOCK:
        PRELOAD_STATE.update(updates)


def _preload_default_t2i_worker(trigger: str) -> None:
    started_at = utc_now()
    _set_preload_state(
        running=True,
        last_trigger=trigger,
        last_started_at=started_at,
        last_finished_at=None,
        last_error=None,
        last_model=None,
    )
    error_text: Optional[str] = None
    model_ref = ""
    try:
        settings = settings_store.get()
        ensure_runtime_dirs(settings)
        model_ref = str(settings.get("defaults", {}).get("text2image_model", "")).strip()
        if not model_ref:
            raise RuntimeError("default text2image model is empty")
        LOGGER.info("preload start trigger=%s task=text-to-image model=%s", trigger, model_ref)
        get_pipeline("text-to-image", model_ref, settings)
        LOGGER.info("preload done trigger=%s task=text-to-image model=%s", trigger, model_ref)
    except Exception as exc:
        error_text = str(exc)
        LOGGER.exception("preload failed trigger=%s task=text-to-image model=%s", trigger, model_ref or "(empty)")
    finally:
        _set_preload_state(
            running=False,
            last_finished_at=utc_now(),
            last_error=error_text,
            last_model=model_ref or None,
        )


def start_preload_default_t2i(trigger: str) -> bool:
    with PRELOAD_LOCK:
        if bool(PRELOAD_STATE.get("running")):
            return False
        PRELOAD_STATE["running"] = True
    worker = threading.Thread(target=_preload_default_t2i_worker, args=(trigger,), daemon=True)
    worker.start()
    return True


def export_video_with_fallback(frames: Any, output_path: Path, fps: int) -> str:
    """
    Prefer AMD AMF hardware encoding on Windows to reduce CPU load.
    Software encoding is intentionally disabled to avoid hidden CPU fallback during diagnostics.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if os.name == "nt":
        try:
            import numpy as np
            import imageio.v2 as imageio  # type: ignore

            def normalize_frame_for_encoder(frame: Any) -> Any:
                arr = np.asarray(frame)
                if arr.ndim == 3 and arr.shape[0] in (1, 3, 4) and arr.shape[-1] not in (1, 3, 4):
                    arr = np.transpose(arr, (1, 2, 0))
                if arr.ndim == 2:
                    arr = np.stack([arr, arr, arr], axis=-1)
                if arr.ndim != 3:
                    raise RuntimeError(f"unsupported frame ndim={arr.ndim}")
                if arr.shape[-1] == 4:
                    arr = arr[..., :3]
                if arr.shape[-1] == 1:
                    arr = np.repeat(arr, 3, axis=-1)
                if arr.dtype != np.uint8:
                    if np.issubdtype(arr.dtype, np.floating):
                        min_v = float(np.nanmin(arr))
                        max_v = float(np.nanmax(arr))
                        if min_v >= -0.01 and max_v <= 1.01:
                            arr = np.clip(arr, 0.0, 1.0) * 255.0
                        else:
                            arr = np.clip(arr, 0.0, 255.0)
                    else:
                        arr = np.clip(arr, 0, 255)
                    arr = arr.astype(np.uint8)
                return arr

            frame_count = 0
            sample_shape: Optional[tuple[int, ...]] = None
            sample_dtype: Optional[str] = None
            normalized_frames: list[Any] = []
            for frame in frames:
                norm = normalize_frame_for_encoder(frame)
                if frame_count == 0:
                    sample_shape = tuple(int(v) for v in norm.shape)
                    sample_dtype = str(norm.dtype)
                normalized_frames.append(norm)
                frame_count += 1

            if frame_count == 0:
                raise RuntimeError("no frames were generated to encode")

            codecs = ("h264_amf", "hevc_amf")
            for codec in codecs:
                try:
                    writer = imageio.get_writer(
                        str(output_path),
                        format="FFMPEG",
                        mode="I",
                        fps=int(fps),
                        codec=codec,
                        macro_block_size=1,
                        ffmpeg_log_level="error",
                        ffmpeg_params=["-pix_fmt", "yuv420p"],
                    )
                    try:
                        for frame in normalized_frames:
                            writer.append_data(frame)
                    finally:
                        writer.close()
                    LOGGER.info(
                        "video encoded with hardware codec=%s path=%s frames=%s fps=%s sample_shape=%s sample_dtype=%s",
                        codec,
                        str(output_path),
                        frame_count,
                        fps,
                        sample_shape,
                        sample_dtype,
                    )
                    return codec
                except Exception:
                    LOGGER.warning(
                        "hardware encode failed codec=%s path=%s frames=%s fps=%s sample_shape=%s sample_dtype=%s",
                        codec,
                        str(output_path),
                        frame_count,
                        fps,
                        sample_shape,
                        sample_dtype,
                        exc_info=True,
                    )
                    if output_path.exists():
                        output_path.unlink(missing_ok=True)
                    continue
        except Exception:
            LOGGER.debug("hardware video encode path unavailable", exc_info=True)

    raise RuntimeError(
        "Hardware video encoder is unavailable. CPU fallback is disabled for root-cause diagnostics."
    )


def call_with_supported_kwargs(pipe: Any, kwargs: Dict[str, Any]) -> Any:
    signature = inspect.signature(pipe.__call__)
    parameters = signature.parameters
    accepts_var_kwargs = any(param.kind == inspect.Parameter.VAR_KEYWORD for param in parameters.values())
    explicit_names = set(parameters.keys())
    if accepts_var_kwargs:
        filtered = {k: v for k, v in kwargs.items() if v is not None}
    else:
        filtered = {k: v for k, v in kwargs.items() if k in explicit_names and v is not None}
    if LOGGER.isEnabledFor(logging.DEBUG):
        dropped = [key for key, value in kwargs.items() if value is None or (not accepts_var_kwargs and key not in explicit_names)]
        LOGGER.debug(
            "pipeline call kwargs pipe=%s accept_var_kwargs=%s provided=%s passed=%s dropped=%s",
            type(pipe).__name__,
            accepts_var_kwargs,
            len(kwargs),
            len(filtered),
            dropped[:20],
        )
    if not accepts_var_kwargs:
        return pipe(**filtered)
    max_retry = 4
    current_kwargs = dict(filtered)
    for _ in range(max_retry):
        try:
            return pipe(**current_kwargs)
        except TypeError as exc:
            # Some wrappers expose **kwargs but still reject unknown keys internally.
            if "unexpected keyword argument" not in str(exc):
                raise
            unknown_match = re.search(r"unexpected keyword argument '([^']+)'", str(exc))
            unknown_key = unknown_match.group(1) if unknown_match else ""
            if not unknown_key or unknown_key not in current_kwargs:
                raise
            LOGGER.debug("pipeline call retry after dropping kwarg pipe=%s dropped=%s", type(pipe).__name__, unknown_key)
            current_kwargs.pop(unknown_key, None)
    return pipe(**current_kwargs)


def build_step_progress_kwargs(
    pipe: Any,
    task_id: str,
    num_inference_steps: int,
    start_progress: float,
    end_progress: float,
    message: str,
) -> Dict[str, Any]:
    total = max(int(num_inference_steps), 1)
    span = max(0.0, float(end_progress) - float(start_progress))
    state = {"last_step": 0}

    def publish(step_index: int) -> None:
        step = max(1, min(int(step_index) + 1, total))
        if step <= int(state["last_step"]):
            return
        state["last_step"] = step
        ratio = step / total
        progress = float(start_progress) + span * ratio
        update_task(task_id, progress=progress, message=f"{message} ({step}/{total})")
        LOGGER.debug("task progress step task_id=%s message=%s step=%s total=%s progress=%.4f", task_id, message, step, total, progress)

    def callback_legacy(step: int, _timestep: Any, _latents: Any) -> None:
        publish(step)

    def callback_on_step_end(_pipe: Any, step: int, _timestep: Any, callback_kwargs: Dict[str, Any]) -> Dict[str, Any]:
        publish(step)
        return callback_kwargs

    signature = inspect.signature(pipe.__call__)
    accepted = set(signature.parameters.keys())
    accepts_var_kwargs = any(param.kind == inspect.Parameter.VAR_KEYWORD for param in signature.parameters.values())
    mode = "none"
    if "callback_on_step_end" in accepted:
        payload: Dict[str, Any] = {
            "callback_on_step_end": callback_on_step_end,
        }
        mode = "callback_on_step_end"
        LOGGER.debug("step callback mode task_id=%s mode=%s", task_id, mode)
        return payload
    if accepts_var_kwargs:
        mode = "callback_on_step_end(var_kwargs)"
        LOGGER.debug("step callback mode task_id=%s mode=%s", task_id, mode)
        return {"callback_on_step_end": callback_on_step_end}
    if "callback" in accepted:
        payload = {"callback": callback_legacy}
        if "callback_steps" in accepted:
            payload["callback_steps"] = 1
        mode = "callback"
        LOGGER.debug("step callback mode task_id=%s mode=%s", task_id, mode)
        return payload
    LOGGER.debug("step callback mode task_id=%s mode=%s", task_id, mode)
    return {}


def resolve_preview_url(model: Any) -> Optional[str]:
    model_id = getattr(model, "id", None)
    if not model_id:
        return None

    def build_url(file_name: str) -> str:
        return f"https://huggingface.co/{quote(model_id, safe='/')}/resolve/main/{quote(file_name, safe='/')}"

    card_data = getattr(model, "cardData", None)
    if isinstance(card_data, dict):
        thumbnail = card_data.get("thumbnail")
        if isinstance(thumbnail, str) and thumbnail.strip():
            thumb = thumbnail.strip()
            if thumb.startswith("http://") or thumb.startswith("https://"):
                return thumb
            return build_url(thumb.lstrip("./"))

    candidates = []
    siblings = getattr(model, "siblings", None) or []
    for sibling in siblings:
        file_name = None
        if isinstance(sibling, dict):
            file_name = sibling.get("rfilename")
        else:
            file_name = getattr(sibling, "rfilename", None)
        if isinstance(file_name, str) and file_name:
            candidates.append(file_name)

    priority = [
        "thumbnail.png",
        "thumbnail.jpg",
        "thumbnail.jpeg",
        "preview.png",
        "preview.jpg",
        "preview.jpeg",
    ]

    lower_map = {name.lower(): name for name in candidates}
    for name in priority:
        if name in lower_map:
            return build_url(lower_map[name])

    for name in candidates:
        lower = name.lower()
        if ("thumbnail" in lower or "preview" in lower) and lower.endswith((".png", ".jpg", ".jpeg", ".webp")):
            return build_url(name)

    return None


def find_local_preview_relpath(model_dir: Path, models_dir: Path) -> Optional[str]:
    priority = [
        "thumbnail.png",
        "thumbnail.jpg",
        "thumbnail.jpeg",
        "preview.png",
        "preview.jpg",
        "preview.jpeg",
        "preview.webp",
    ]
    files = []
    for candidate in priority:
        path = model_dir / candidate
        if path.exists() and path.is_file():
            files.append(path)
    if not files:
        for path in model_dir.iterdir():
            if path.is_file() and path.suffix.lower() in (".png", ".jpg", ".jpeg", ".webp"):
                lower = path.name.lower()
                if "thumbnail" in lower or "preview" in lower:
                    files.append(path)
    if not files:
        return None
    rel = files[0].resolve().relative_to(models_dir.resolve())
    return str(rel).replace("\\", "/")


def load_local_model_index(model_dir: Path) -> Optional[Dict[str, Any]]:
    index_path = model_dir / "model_index.json"
    if not index_path.exists() or not index_path.is_file():
        return None
    try:
        payload = json.loads(index_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    return payload


def extract_base_model_hint(model_dir: Path, model_index: Optional[Dict[str, Any]]) -> Optional[str]:
    adapter_config = parse_lora_adapter_config(model_dir)
    adapter_base = str(adapter_config.get("base_model_name_or_path") or "").strip()
    if adapter_base:
        return adapter_base
    if isinstance(model_index, dict):
        for key in ("base_model", "base_model_name_or_path", "_name_or_path"):
            value = str(model_index.get(key) or "").strip()
            if value and value not in (".", "null", "None"):
                return value
    # Fallback: estimate lineage from local repository name / class hint.
    class_hint = str((model_index or {}).get("_class_name") or "").strip()
    repo_hint = desanitize_repo_id(model_dir.name)
    estimated = infer_base_model_label(repo_hint, class_hint)
    if estimated != "Other":
        return estimated
    return None


def detect_local_model_meta(model_dir: Path) -> LocalModelMeta:
    model_index = load_local_model_index(model_dir)
    class_name = str((model_index or {}).get("_class_name") or "").strip()
    is_lora = is_local_lora_dir(model_dir)
    is_vae = is_local_vae_dir(model_dir)
    if is_lora and not class_name:
        class_name = "LoRA"
    compatible_tasks: list[str] = []
    for task, allowed in SUPPORTED_LOCAL_PIPELINE_CLASSES.items():
        if class_name in allowed:
            compatible_tasks.append(task)
    base_model = extract_base_model_hint(model_dir, model_index)
    return LocalModelMeta(
        class_name=class_name,
        base_model=base_model,
        compatible_tasks=compatible_tasks,
        is_lora=is_lora,
        is_vae=is_vae,
    )


def is_local_model_compatible(task: Literal["text-to-image", "image-to-image", "text-to-video", "image-to-video"], model_dir: Path) -> bool:
    model_index = load_local_model_index(model_dir)
    if not model_index:
        return False
    class_name = str(model_index.get("_class_name") or "").strip()
    allowed = SUPPORTED_LOCAL_PIPELINE_CLASSES.get(task, set())
    return class_name in allowed


def extract_model_size_bytes_from_siblings(siblings: Any) -> Optional[int]:
    total = 0
    found = False
    for sibling in siblings or []:
        size_value: Any = None
        lfs_info: Any = None
        if isinstance(sibling, dict):
            size_value = sibling.get("size")
            lfs_info = sibling.get("lfs")
        else:
            size_value = getattr(sibling, "size", None)
            lfs_info = getattr(sibling, "lfs", None)
        if not isinstance(size_value, int) or size_value <= 0:
            if isinstance(lfs_info, dict):
                lfs_size = lfs_info.get("size")
                if isinstance(lfs_size, int) and lfs_size > 0:
                    size_value = lfs_size
        if isinstance(size_value, int) and size_value > 0:
            total += size_value
            found = True
    return total if found and total > 0 else None


def get_remote_model_size_bytes(api: HfApi, model: Any, repo_id: str) -> Optional[int]:
    size_bytes = extract_model_size_bytes_from_siblings(getattr(model, "siblings", None))
    if size_bytes is not None:
        return size_bytes

    now = time.time()
    with MODEL_SIZE_CACHE_LOCK:
        cached = MODEL_SIZE_CACHE.get(repo_id)
        if cached and (now - cached["ts"]) < MODEL_SIZE_CACHE_TTL_SEC:
            cached_size = cached.get("size_bytes")
            return cached_size if isinstance(cached_size, int) else None

    fetched_size: Optional[int] = None
    try:
        info = api.model_info(repo_id=repo_id, files_metadata=True)
        fetched_size = extract_model_size_bytes_from_siblings(getattr(info, "siblings", None))
    except Exception:
        fetched_size = None

    with MODEL_SIZE_CACHE_LOCK:
        MODEL_SIZE_CACHE[repo_id] = {"size_bytes": fetched_size, "ts": now}
    return fetched_size


def infer_base_model_label(*parts: Any) -> str:
    text = " ".join(str(part or "") for part in parts).strip().lower()
    if not text:
        return "Other"
    if "stable-diffusion-xl" in text or "sdxl" in text or re.search(r"\bxl\b", text):
        return "StableDiffusion XL"
    if "stable-diffusion-2-1" in text or "stable diffusion 2.1" in text or re.search(r"\b2\.1\b", text):
        return "StableDiffusion 2.1"
    if "stable-diffusion-2" in text or "stable diffusion 2" in text or re.search(r"\bsd2\b", text):
        return "StableDiffusion 2.x"
    if "stable-diffusion-1-5" in text or "stable diffusion 1.5" in text or re.search(r"\b1\.5\b", text):
        return "StableDiffusion 1.5"
    if "stable-diffusion-1" in text or "stable diffusion 1" in text or re.search(r"\bsd1\b", text):
        return "StableDiffusion 1.x"
    if "flux" in text:
        return "FLUX"
    if "pixart" in text:
        return "PixArt"
    if "auraflow" in text:
        return "AuraFlow"
    if "i2vgen" in text:
        return "I2VGenXL"
    if "texttovideosd" in text or "text-to-video-ms" in text or "text-to-video" in text:
        return "TextToVideoSD"
    if "wan" in text:
        return "Wan"
    return "Other"


def normalize_base_model_filter(value: str) -> str:
    raw = str(value or "").strip()
    if not raw:
        return ""
    lowered = raw.lower()
    if lowered in ("all", "any", "*"):
        return ""
    if lowered == "other":
        return "Other"
    inferred = infer_base_model_label(raw)
    if inferred != "Other":
        return inferred
    return raw


def matches_base_model_filter(candidate: str, selected: str) -> bool:
    normalized_selected = normalize_base_model_filter(selected)
    if not normalized_selected:
        return True
    candidate_label = str(candidate or "").strip() or "Other"
    if normalized_selected == "Other":
        return infer_base_model_label(candidate_label) == "Other"
    return candidate_label.lower() == normalized_selected.lower() or infer_base_model_label(candidate_label) == normalized_selected


def search_hf_models(
    task: Literal["text-to-image", "image-to-image", "text-to-video", "image-to-video"],
    query: str,
    limit: int,
    token: Optional[str],
) -> list[Dict[str, Any]]:
    api = HfApi(token=token)
    capped_limit = min(max(limit, 1), 50)
    normalized_query = query.strip()
    if normalized_query:
        found = api.list_models(
            search=normalized_query,
            sort="downloads",
            direction=-1,
            limit=capped_limit * 6,
            full=True,
            cardData=True,
        )
    else:
        found = api.list_models(
            filter=task,
            sort="downloads",
            direction=-1,
            limit=capped_limit * 3,
            full=True,
            cardData=True,
        )
    results: list[Dict[str, Any]] = []
    for model in found:
        pipeline_tag = getattr(model, "pipeline_tag", None)
        tags = getattr(model, "tags", []) or []
        matches_task = pipeline_tag == task or task in tags
        if not matches_task:
            continue
        model_id = getattr(model, "id", "")
        if not model_id:
            continue
        card_data = getattr(model, "cardData", None)
        card_hints: list[str] = []
        if isinstance(card_data, dict):
            for key in ("base_model", "baseModel", "base_models", "baseModels", "model_type", "modelType"):
                value = card_data.get(key)
                if isinstance(value, list):
                    card_hints.extend(str(v) for v in value)
                elif value:
                    card_hints.append(str(value))
        base_model = infer_base_model_label(model_id, pipeline_tag, " ".join(str(tag) for tag in tags), " ".join(card_hints))
        size_bytes = get_remote_model_size_bytes(api, model, model_id)
        results.append(
            {
                "id": model_id,
                "pipeline_tag": pipeline_tag,
                "downloads": getattr(model, "downloads", None),
                "likes": getattr(model, "likes", None),
                "private": getattr(model, "private", False),
                "size_bytes": size_bytes,
                "model_url": f"https://huggingface.co/{quote(model_id, safe='/')}",
                "preview_url": resolve_preview_url(model),
                "base_model": base_model,
                "source": "huggingface",
                "download_supported": True,
            }
        )
        if len(results) >= capped_limit:
            break

    if not results and not normalized_query:
        fallback = api.list_models(
            search=task,
            sort="downloads",
            direction=-1,
            limit=capped_limit * 4,
            full=True,
            cardData=True,
        )
        for model in fallback:
            pipeline_tag = getattr(model, "pipeline_tag", None)
            tags = getattr(model, "tags", []) or []
            matches_task = pipeline_tag == task or task in tags
            if not matches_task:
                continue
            model_id = getattr(model, "id", "")
            if not model_id:
                continue
            card_data = getattr(model, "cardData", None)
            card_hints: list[str] = []
            if isinstance(card_data, dict):
                for key in ("base_model", "baseModel", "base_models", "baseModels", "model_type", "modelType"):
                    value = card_data.get(key)
                    if isinstance(value, list):
                        card_hints.extend(str(v) for v in value)
                    elif value:
                        card_hints.append(str(value))
            base_model = infer_base_model_label(model_id, pipeline_tag, " ".join(str(tag) for tag in tags), " ".join(card_hints))
            size_bytes = get_remote_model_size_bytes(api, model, model_id)
            results.append(
                {
                    "id": model_id,
                    "pipeline_tag": pipeline_tag,
                    "downloads": getattr(model, "downloads", None),
                    "likes": getattr(model, "likes", None),
                    "private": getattr(model, "private", False),
                    "size_bytes": size_bytes,
                    "model_url": f"https://huggingface.co/{quote(model_id, safe='/')}",
                    "preview_url": resolve_preview_url(model),
                    "base_model": base_model,
                    "source": "huggingface",
                    "download_supported": True,
                }
            )
            if len(results) >= capped_limit:
                break

    return results


def civitai_task_model_types(task: str) -> list[str]:
    if task in ("text-to-image", "image-to-image"):
        return ["Checkpoint"]
    return []


def safe_int(value: Any) -> Optional[int]:
    try:
        parsed = int(value)
    except Exception:
        return None
    return parsed if parsed >= 0 else None


def directory_size_bytes(root: Path) -> int:
    if not root.exists():
        return 0
    total = 0
    for entry in root.rglob("*"):
        if not entry.is_file():
            continue
        try:
            total += int(entry.stat().st_size)
        except OSError:
            continue
    return total


def civitai_request_json(path: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    query = urlencode(params or {}, doseq=True)
    url = f"{CIVITAI_API_BASE}/{path.lstrip('/')}"
    if query:
        url = f"{url}?{query}"
    req = UrlRequest(url, headers={"User-Agent": "ROCm-VideoGen/1.0"})
    with contextlib.closing(urlopen(req, timeout=20)) as resp:
        payload = json.loads(resp.read().decode("utf-8"))
    return payload if isinstance(payload, dict) else {}


def parse_civitai_model_id(repo_id: str) -> Optional[int]:
    normalized = str(repo_id or "").strip()
    if not normalized.lower().startswith("civitai/"):
        return None
    suffix = normalized.split("/", 1)[1].strip()
    if not suffix:
        return None
    match = re.match(r"^(\d+)", suffix)
    if not match:
        return None
    return safe_int(match.group(1))


def sanitize_download_filename(value: str, fallback: str) -> str:
    raw = str(value or "").strip()
    if not raw:
        raw = fallback
    safe = re.sub(r'[<>:"/\\|?*\x00-\x1F]+', "_", raw).strip(" .")
    return safe or fallback


def extract_civitai_primary_file(model_payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    versions = model_payload.get("modelVersions", [])
    if not isinstance(versions, list):
        return None
    for version in versions:
        if not isinstance(version, dict):
            continue
        files = version.get("files", [])
        if not isinstance(files, list) or not files:
            continue
        model_files: list[Dict[str, Any]] = []
        fallback_files: list[Dict[str, Any]] = []
        for file_entry in files:
            if not isinstance(file_entry, dict):
                continue
            download_url = str(file_entry.get("downloadUrl") or "").strip()
            if not download_url:
                continue
            kind = str(file_entry.get("type") or "").strip().lower()
            if kind in ("model", "checkpoint"):
                model_files.append(file_entry)
            else:
                fallback_files.append(file_entry)
        chosen_file = model_files[0] if model_files else (fallback_files[0] if fallback_files else None)
        if not chosen_file:
            continue
        merged = dict(chosen_file)
        merged["version_id"] = version.get("id")
        merged["version_name"] = version.get("name")
        return merged
    return None


def search_civitai_models(task: Literal["text-to-image", "image-to-image", "text-to-video", "image-to-video"], query: str, limit: int) -> list[Dict[str, Any]]:
    model_types = civitai_task_model_types(task)
    if not model_types:
        return []

    capped_limit = min(max(limit, 1), 50)
    params: Dict[str, Any] = {
        "limit": capped_limit,
        "nsfw": "false",
        "sort": "Most Downloaded",
        "period": "AllTime",
    }
    if query.strip():
        params["query"] = query.strip()
    for model_type in model_types:
        params.setdefault("types", [])
        params["types"].append(model_type)
    try:
        payload = civitai_request_json("models", params)
    except (HTTPError, URLError, TimeoutError, json.JSONDecodeError, UnicodeDecodeError):
        LOGGER.warning("civitai search failed task=%s query=%s", task, query, exc_info=True)
        return []

    items = payload.get("items", []) if isinstance(payload, dict) else []
    results: list[Dict[str, Any]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        model_id = safe_int(item.get("id"))
        if not model_id:
            continue
        model_name = str(item.get("name") or f"model-{model_id}")
        model_type = str(item.get("type") or "")
        stats = item.get("stats", {}) if isinstance(item.get("stats"), dict) else {}
        downloads = safe_int(stats.get("downloadCount"))
        likes = safe_int(stats.get("favoriteCount")) or safe_int(stats.get("thumbsUpCount"))
        preview_url = None
        size_bytes = None
        has_download_file = False
        base_model_raw = ""
        model_versions = item.get("modelVersions", [])
        if isinstance(model_versions, list) and model_versions:
            first_version = model_versions[0] if isinstance(model_versions[0], dict) else {}
            base_model_raw = str(first_version.get("baseModel") or "").strip()
            images = first_version.get("images", [])
            if isinstance(images, list):
                for image in images:
                    if isinstance(image, dict):
                        image_url = str(image.get("url") or "").strip()
                        if image_url:
                            preview_url = image_url
                            break
            files = first_version.get("files", [])
            if isinstance(files, list) and files:
                first_file = files[0] if isinstance(files[0], dict) else {}
                for candidate_file in files:
                    if isinstance(candidate_file, dict) and str(candidate_file.get("downloadUrl") or "").strip():
                        has_download_file = True
                        break
                size_kb = first_file.get("sizeKB")
                if isinstance(size_kb, (int, float)) and size_kb > 0:
                    size_bytes = int(float(size_kb) * 1024)
        if not base_model_raw:
            base_model_raw = str(item.get("baseModel") or "").strip()
        base_model = infer_base_model_label(base_model_raw, model_name, model_type, task)
        result_id = f"civitai/{model_id}"
        results.append(
            {
                "id": result_id,
                "pipeline_tag": task,
                "downloads": downloads,
                "likes": likes,
                "private": False,
                "size_bytes": size_bytes,
                "model_url": f"https://civitai.com/models/{model_id}",
                "preview_url": preview_url,
                "source": "civitai",
                "download_supported": has_download_file,
                "label": f"[civitai] {model_name}",
                "type": model_type,
                "base_model": base_model,
            }
        )
        if len(results) >= capped_limit:
            break
    return results


def stream_http_download(
    url: str,
    destination: Path,
    task_id: str,
    progress_message: str,
    total_bytes_hint: Optional[int] = None,
) -> int:
    req = UrlRequest(url, headers={"User-Agent": "ROCm-VideoGen/1.0"})
    downloaded = 0
    last_reported = 0
    last_reported_ts = 0.0
    with contextlib.closing(urlopen(req, timeout=60)) as resp, destination.open("wb") as out_file:
        total = safe_int(resp.headers.get("Content-Length")) or safe_int(total_bytes_hint)
        if total and total > 0:
            update_task(task_id, progress=0.0, message=progress_message, downloaded_bytes=0, total_bytes=total)
        while True:
            chunk = resp.read(DOWNLOAD_STREAM_CHUNK_BYTES)
            if not chunk:
                break
            out_file.write(chunk)
            downloaded += len(chunk)
            now = time.time()
            should_report = (downloaded - last_reported >= (2 * DOWNLOAD_STREAM_CHUNK_BYTES)) or ((now - last_reported_ts) >= 1.0)
            if should_report:
                last_reported = downloaded
                last_reported_ts = now
                if total and total > 0:
                    ratio = min(downloaded / total, 1.0)
                    update_task(
                        task_id,
                        progress=min(ratio, 0.99),
                        message=progress_message,
                        downloaded_bytes=downloaded,
                        total_bytes=total,
                    )
                else:
                    update_task(
                        task_id,
                        progress=min(0.95, (downloaded / (200 * 1024 * 1024))),
                        message=progress_message,
                        downloaded_bytes=downloaded,
                        total_bytes=None,
                    )
    if total and total > 0:
        update_task(task_id, progress=min(0.99, downloaded / total), message=progress_message, downloaded_bytes=downloaded, total_bytes=total)
    else:
        update_task(task_id, progress=min(0.95, (downloaded / (200 * 1024 * 1024))), message=progress_message, downloaded_bytes=downloaded, total_bytes=None)
    return downloaded


class Text2ImageRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=2000)
    negative_prompt: str = Field(default="", max_length=2000)
    model_id: Optional[str] = None
    lora_id: Optional[str] = None
    lora_ids: list[str] = Field(default_factory=list)
    lora_scale: float = Field(default=1.0, ge=0.0, le=2.0)
    vae_id: Optional[str] = None
    num_inference_steps: int = Field(default=30, ge=1, le=120)
    guidance_scale: float = Field(default=7.5, ge=0.0, le=30.0)
    width: int = Field(default=512, ge=128, le=2048)
    height: int = Field(default=512, ge=128, le=2048)
    seed: Optional[int] = Field(default=None, ge=0)


class Text2VideoRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=2000)
    negative_prompt: str = Field(default="", max_length=2000)
    model_id: Optional[str] = None
    lora_id: Optional[str] = None
    lora_ids: list[str] = Field(default_factory=list)
    lora_scale: float = Field(default=1.0, ge=0.0, le=2.0)
    num_inference_steps: int = Field(default=30, ge=1, le=120)
    num_frames: int = Field(default=16, ge=8, le=128)
    guidance_scale: float = Field(default=9.0, ge=0.0, le=30.0)
    fps: int = Field(default=8, ge=1, le=60)
    seed: Optional[int] = Field(default=None, ge=0)
    backend: Literal["auto", "cuda", "npu"] = "auto"


class Image2ImageRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=2000)
    negative_prompt: str = Field(default="", max_length=2000)
    model_id: Optional[str] = None
    lora_id: Optional[str] = None
    lora_ids: list[str] = Field(default_factory=list)
    lora_scale: float = Field(default=1.0, ge=0.0, le=2.0)
    vae_id: Optional[str] = None
    num_inference_steps: int = Field(default=30, ge=1, le=120)
    guidance_scale: float = Field(default=7.5, ge=0.0, le=30.0)
    strength: float = Field(default=0.8, ge=0.0, le=1.0)
    width: int = Field(default=512, ge=128, le=2048)
    height: int = Field(default=512, ge=128, le=2048)
    seed: Optional[int] = Field(default=None, ge=0)


class DownloadRequest(BaseModel):
    repo_id: str = Field(min_length=3)
    revision: Optional[str] = None
    target_dir: Optional[str] = None


class DeleteLocalModelRequest(BaseModel):
    model_name: str = Field(min_length=1)
    base_dir: Optional[str] = None


class DeleteOutputRequest(BaseModel):
    file_name: str = Field(min_length=1)


class ClearHfCacheRequest(BaseModel):
    dry_run: bool = False


def text2image_worker(task_id: str, payload: Text2ImageRequest) -> None:
    settings = settings_store.get()
    ensure_runtime_dirs(settings)
    model_ref = payload.model_id or settings["defaults"]["text2image_model"]
    lora_refs = collect_lora_refs(payload.lora_id, payload.lora_ids)
    vae_ref = (payload.vae_id or "").strip()
    LOGGER.info(
        "text2image start task_id=%s model=%s loras=%s lora_scale=%s vae=%s steps=%s guidance=%s size=%sx%s seed=%s",
        task_id,
        model_ref,
        ",".join(lora_refs) if lora_refs else "(none)",
        payload.lora_scale,
        vae_ref or "(none)",
        payload.num_inference_steps,
        payload.guidance_scale,
        payload.width,
        payload.height,
        payload.seed,
    )
    try:
        update_task(task_id, status="running", progress=0.05, message="Loading model")
        pipe = get_pipeline("text-to-image", model_ref, settings)
        device, dtype = get_device_and_dtype()
        update_task(task_id, progress=0.1, message="Loading model")
        apply_vae_to_pipeline(pipe, vae_ref, settings, device=device, dtype=dtype)
        update_task(task_id, progress=0.15, message="Applying LoRA")
        apply_loras_to_pipeline(pipe, lora_refs, payload.lora_scale, settings)
        update_task(task_id, progress=0.2, message=f"Generating image (0/{payload.num_inference_steps})")
        generator = None
        if payload.seed is not None:
            gen_device = "cuda" if device == "cuda" else "cpu"
            generator = torch.Generator(device=gen_device).manual_seed(payload.seed)
        step_progress_kwargs = build_step_progress_kwargs(
            pipe=pipe,
            task_id=task_id,
            num_inference_steps=payload.num_inference_steps,
            start_progress=0.2,
            end_progress=0.9,
            message="Generating image",
        )
        LOGGER.info(
            "text2image inference start task_id=%s model=%s steps=%s guidance=%s size=%sx%s callback_keys=%s",
            task_id,
            model_ref,
            payload.num_inference_steps,
            payload.guidance_scale,
            payload.width,
            payload.height,
            ",".join(sorted(step_progress_kwargs.keys())) if step_progress_kwargs else "(none)",
        )
        gen_started = time.perf_counter()
        out = call_with_supported_kwargs(
            pipe,
            {
                "prompt": payload.prompt,
                "negative_prompt": payload.negative_prompt or None,
                "num_inference_steps": payload.num_inference_steps,
                "guidance_scale": payload.guidance_scale,
                "width": payload.width,
                "height": payload.height,
                "generator": generator,
                "output_type": "pil",
                "cross_attention_kwargs": {"scale": payload.lora_scale} if len(lora_refs) == 1 else None,
                **step_progress_kwargs,
            },
        )
        LOGGER.info("text2image inference done task_id=%s elapsed_ms=%.1f", task_id, (time.perf_counter() - gen_started) * 1000)
        images: list[Any]
        raw_images = out.images
        if isinstance(raw_images, list) and raw_images and isinstance(raw_images[0], Image.Image):
            images = raw_images
            update_task(task_id, progress=0.96, message="Postprocessing image")
        else:
            update_task(task_id, progress=0.92, message="Decoding latents")
            decode_started = time.perf_counter()
            log_gpu_memory_stats("before_decode", task_id)
            try:
                with task_progress_heartbeat(
                    task_id,
                    start_progress=0.92,
                    end_progress=0.955,
                    message="Decoding latents",
                    interval_sec=0.5,
                    estimated_duration_sec=12.0,
                ):
                    images = decode_latents_to_pil_images(pipe, raw_images)
            except torch.OutOfMemoryError:
                LOGGER.error(
                    "text2image decode OOM task_id=%s. CPU fallback is disabled for diagnostics.",
                    task_id,
                    exc_info=True,
                )
                log_gpu_memory_stats("decode_oom", task_id)
                raise
            LOGGER.info("text2image decode done task_id=%s elapsed_ms=%.1f", task_id, (time.perf_counter() - decode_started) * 1000)
            log_gpu_memory_stats("after_decode", task_id)
            update_task(task_id, progress=0.96, message="Postprocessing image")
        image = images[0]
        update_task(task_id, progress=0.98, message="Saving png")
        output_name = f"text2image_{task_id}.png"
        output_path = resolve_path(settings["paths"]["outputs_dir"]) / output_name
        with task_progress_heartbeat(
            task_id,
            start_progress=0.98,
            end_progress=0.995,
            message="Saving png",
            interval_sec=0.4,
            estimated_duration_sec=4.0,
        ):
            image.save(str(output_path))
        update_task(
            task_id,
            status="completed",
            progress=1.0,
            message="Done",
            result={"image_file": output_name, "model": model_ref, "loras": lora_refs, "vae": vae_ref or None},
        )
        LOGGER.info("text2image done task_id=%s output=%s", task_id, str(output_path))
    except Exception as exc:
        trace = format_exception_trace()
        LOGGER.exception("text2image failed task_id=%s model=%s diagnostics=%s", task_id, model_ref, runtime_diagnostics())
        update_task(task_id, status="error", message="Generation failed", error=str(exc), error_trace=trace)


def text2video_worker(task_id: str, payload: Text2VideoRequest) -> None:
    settings = settings_store.get()
    ensure_runtime_dirs(settings)
    model_ref = payload.model_id or settings["defaults"]["text2video_model"]
    lora_refs = collect_lora_refs(payload.lora_id, payload.lora_ids)
    effective_backend = resolve_text2video_backend(payload.backend, settings)
    LOGGER.info(
        "text2video start task_id=%s model=%s backend=%s requested_backend=%s loras=%s lora_scale=%s steps=%s frames=%s guidance=%s fps=%s seed=%s",
        task_id,
        model_ref,
        effective_backend,
        payload.backend,
        ",".join(lora_refs) if lora_refs else "(none)",
        payload.lora_scale,
        payload.num_inference_steps,
        payload.num_frames,
        payload.guidance_scale,
        payload.fps,
        payload.seed,
    )
    try:
        if effective_backend == "npu":
            if lora_refs:
                raise RuntimeError("NPU backend for text-to-video does not support LoRA yet.")
            update_task(task_id, status="running", progress=0.05, message="Loading model")
            result = run_text2video_npu_runner(task_id, payload, settings, model_ref)
            update_task(
                task_id,
                status="completed",
                progress=1.0,
                message="Done",
                result={
                    "video_file": result["video_file"],
                    "model": model_ref,
                    "loras": [],
                    "encoder": result.get("encoder"),
                    "backend": "npu",
                    "runner": result.get("runner"),
                },
            )
            LOGGER.info(
                "text2video done task_id=%s backend=npu output=%s runner=%s",
                task_id,
                result["video_file"],
                result.get("runner"),
            )
            return

        update_task(task_id, status="running", progress=0.05, message="Loading model")
        pipe = get_pipeline("text-to-video", model_ref, settings)
        update_task(task_id, progress=0.15, message="Applying LoRA")
        try:
            apply_loras_to_pipeline(pipe, lora_refs, payload.lora_scale, settings)
        except Exception as exc:
            if is_lora_error_compatible_skip(exc):
                LOGGER.warning(
                    "text2video lora skipped as incompatible task_id=%s model=%s loras=%s error=%s",
                    task_id,
                    model_ref,
                    ",".join(lora_refs) if lora_refs else "(none)",
                    str(exc),
                )
                lora_refs = []
                update_task(task_id, progress=0.18, message="Applying LoRA (skipped)")
            else:
                raise
        update_task(task_id, progress=0.35, message=f"Generating frames (0/{payload.num_inference_steps})")
        device, _ = get_device_and_dtype()
        generator = None
        if payload.seed is not None:
            gen_device = "cuda" if device == "cuda" else "cpu"
            generator = torch.Generator(device=gen_device).manual_seed(payload.seed)
        step_progress_kwargs = build_step_progress_kwargs(
            pipe=pipe,
            task_id=task_id,
            num_inference_steps=payload.num_inference_steps,
            start_progress=0.35,
            end_progress=0.88,
            message="Generating frames",
        )
        LOGGER.info(
            "text2video inference start task_id=%s model=%s steps=%s frames=%s guidance=%s callback_keys=%s",
            task_id,
            model_ref,
            payload.num_inference_steps,
            payload.num_frames,
            payload.guidance_scale,
            ",".join(sorted(step_progress_kwargs.keys())) if step_progress_kwargs else "(none)",
        )
        gen_started = time.perf_counter()
        out = call_with_supported_kwargs(
            pipe,
            {
                "prompt": payload.prompt,
                "negative_prompt": payload.negative_prompt or None,
                "num_inference_steps": payload.num_inference_steps,
                "num_frames": payload.num_frames,
                "guidance_scale": payload.guidance_scale,
                "generator": generator,
                "cross_attention_kwargs": {"scale": payload.lora_scale} if len(lora_refs) == 1 else None,
                **step_progress_kwargs,
            },
        )
        LOGGER.info("text2video frame generation done task_id=%s elapsed_ms=%.1f", task_id, (time.perf_counter() - gen_started) * 1000)
        frames = out.frames[0]
        update_task(task_id, progress=0.9, message="Encoding mp4")
        output_name = f"text2video_{task_id}.mp4"
        output_path = resolve_path(settings["paths"]["outputs_dir"]) / output_name
        enc_started = time.perf_counter()
        LOGGER.info("text2video encoding start task_id=%s frames=%s fps=%s output=%s", task_id, len(frames), payload.fps, str(output_path))
        with task_progress_heartbeat(
            task_id,
            start_progress=0.9,
            end_progress=0.99,
            message="Encoding mp4",
            interval_sec=0.5,
            estimated_duration_sec=max(8.0, min(180.0, len(frames) * 0.4)),
        ):
            encoder_name = export_video_with_fallback(frames, output_path, fps=int(payload.fps))
        LOGGER.info("text2video encoding done task_id=%s encoder=%s elapsed_ms=%.1f", task_id, encoder_name, (time.perf_counter() - enc_started) * 1000)
        update_task(
            task_id,
            status="completed",
            progress=1.0,
            message="Done",
            result={"video_file": output_name, "model": model_ref, "loras": lora_refs, "encoder": encoder_name, "backend": "cuda"},
        )
        LOGGER.info("text2video done task_id=%s backend=cuda output=%s", task_id, str(output_path))
    except Exception as exc:
        trace = format_exception_trace()
        LOGGER.exception(
            "text2video failed task_id=%s model=%s backend=%s diagnostics=%s",
            task_id,
            model_ref,
            effective_backend,
            runtime_diagnostics(),
        )
        update_task(task_id, status="error", message="Generation failed", error=str(exc), error_trace=trace)


def image2video_worker(task_id: str, payload: Dict[str, Any]) -> None:
    settings = settings_store.get()
    ensure_runtime_dirs(settings)
    model_ref = payload["model_id"] or settings["defaults"]["image2video_model"]
    lora_refs = collect_lora_refs(payload.get("lora_id"), payload.get("lora_ids") or [])
    lora_scale = float(payload.get("lora_scale") or 1.0)
    image_path = Path(payload["image_path"])
    LOGGER.info(
        "image2video start task_id=%s model=%s loras=%s lora_scale=%s image=%s steps=%s frames=%s guidance=%s fps=%s size=%sx%s seed=%s",
        task_id,
        model_ref,
        ",".join(lora_refs) if lora_refs else "(none)",
        lora_scale,
        str(image_path),
        payload.get("num_inference_steps"),
        payload.get("num_frames"),
        payload.get("guidance_scale"),
        payload.get("fps"),
        payload.get("width"),
        payload.get("height"),
        payload.get("seed"),
    )
    try:
        update_task(task_id, status="running", progress=0.05, message="Loading model")
        pipe = get_pipeline("image-to-video", model_ref, settings)
        update_task(task_id, progress=0.15, message="Applying LoRA")
        try:
            apply_loras_to_pipeline(pipe, lora_refs, lora_scale, settings)
        except Exception as exc:
            if is_lora_error_compatible_skip(exc):
                LOGGER.warning(
                    "image2video lora skipped as incompatible task_id=%s model=%s loras=%s error=%s",
                    task_id,
                    model_ref,
                    ",".join(lora_refs) if lora_refs else "(none)",
                    str(exc),
                )
                lora_refs = []
                update_task(task_id, progress=0.18, message="Applying LoRA (skipped)")
            else:
                raise
        update_task(task_id, progress=0.35, message="Preparing image")
        image = Image.open(image_path).convert("RGB")
        width = int(payload["width"])
        height = int(payload["height"])
        if width > 0 and height > 0:
            image = image.resize((width, height))
        device, _ = get_device_and_dtype()
        generator = None
        if payload["seed"] is not None:
            gen_device = "cuda" if device == "cuda" else "cpu"
            generator = torch.Generator(device=gen_device).manual_seed(int(payload["seed"]))
        step_count = int(payload["num_inference_steps"])
        update_task(task_id, progress=0.45, message=f"Generating frames (0/{step_count})")
        step_progress_kwargs = build_step_progress_kwargs(
            pipe=pipe,
            task_id=task_id,
            num_inference_steps=step_count,
            start_progress=0.45,
            end_progress=0.88,
            message="Generating frames",
        )
        LOGGER.info(
            "image2video inference start task_id=%s model=%s steps=%s frames=%s guidance=%s size=%sx%s callback_keys=%s",
            task_id,
            model_ref,
            payload["num_inference_steps"],
            payload["num_frames"],
            payload["guidance_scale"],
            width,
            height,
            ",".join(sorted(step_progress_kwargs.keys())) if step_progress_kwargs else "(none)",
        )
        gen_started = time.perf_counter()
        out = call_with_supported_kwargs(
            pipe,
            {
                "prompt": payload["prompt"],
                "negative_prompt": payload["negative_prompt"] or None,
                "image": image,
                "height": height,
                "width": width,
                "target_fps": int(payload["fps"]),
                "num_inference_steps": int(payload["num_inference_steps"]),
                "num_frames": int(payload["num_frames"]),
                "guidance_scale": float(payload["guidance_scale"]),
                "generator": generator,
                "cross_attention_kwargs": {"scale": lora_scale} if len(lora_refs) == 1 else None,
                **step_progress_kwargs,
            },
        )
        LOGGER.info("image2video frame generation done task_id=%s elapsed_ms=%.1f", task_id, (time.perf_counter() - gen_started) * 1000)
        frames = out.frames[0]
        update_task(task_id, progress=0.9, message="Encoding mp4")
        output_name = f"image2video_{task_id}.mp4"
        output_path = resolve_path(settings["paths"]["outputs_dir"]) / output_name
        enc_started = time.perf_counter()
        LOGGER.info(
            "image2video encoding start task_id=%s frames=%s fps=%s output=%s",
            task_id,
            len(frames),
            payload["fps"],
            str(output_path),
        )
        with task_progress_heartbeat(
            task_id,
            start_progress=0.9,
            end_progress=0.99,
            message="Encoding mp4",
            interval_sec=0.5,
            estimated_duration_sec=max(8.0, min(180.0, len(frames) * 0.4)),
        ):
            encoder_name = export_video_with_fallback(frames, output_path, fps=int(payload["fps"]))
        LOGGER.info("image2video encoding done task_id=%s encoder=%s elapsed_ms=%.1f", task_id, encoder_name, (time.perf_counter() - enc_started) * 1000)
        update_task(
            task_id,
            status="completed",
            progress=1.0,
            message="Done",
            result={"video_file": output_name, "model": model_ref, "loras": lora_refs, "encoder": encoder_name},
        )
        LOGGER.info("image2video done task_id=%s output=%s", task_id, str(output_path))
    except Exception as exc:
        trace = format_exception_trace()
        LOGGER.exception("image2video failed task_id=%s model=%s diagnostics=%s", task_id, model_ref, runtime_diagnostics())
        update_task(task_id, status="error", message="Generation failed", error=str(exc), error_trace=trace)
    finally:
        if image_path.exists():
            image_path.unlink(missing_ok=True)


def image2image_worker(task_id: str, payload: Dict[str, Any]) -> None:
    settings = settings_store.get()
    ensure_runtime_dirs(settings)
    model_ref = payload["model_id"] or settings["defaults"]["image2image_model"]
    lora_refs = collect_lora_refs(payload.get("lora_id"), payload.get("lora_ids") or [])
    lora_scale = float(payload.get("lora_scale") or 1.0)
    vae_ref = str(payload.get("vae_id") or "").strip()
    image_path = Path(payload["image_path"])
    LOGGER.info(
        "image2image start task_id=%s model=%s loras=%s lora_scale=%s vae=%s image=%s steps=%s guidance=%s strength=%s size=%sx%s seed=%s",
        task_id,
        model_ref,
        ",".join(lora_refs) if lora_refs else "(none)",
        lora_scale,
        vae_ref or "(none)",
        str(image_path),
        payload.get("num_inference_steps"),
        payload.get("guidance_scale"),
        payload.get("strength"),
        payload.get("width"),
        payload.get("height"),
        payload.get("seed"),
    )
    try:
        update_task(task_id, status="running", progress=0.05, message="Loading model")
        pipe = get_pipeline("image-to-image", model_ref, settings)
        device, dtype = get_device_and_dtype()
        update_task(task_id, progress=0.1, message="Loading model")
        apply_vae_to_pipeline(pipe, vae_ref, settings, device=device, dtype=dtype)
        update_task(task_id, progress=0.15, message="Applying LoRA")
        apply_loras_to_pipeline(pipe, lora_refs, lora_scale, settings)
        update_task(task_id, progress=0.35, message="Preparing image")
        image = Image.open(image_path).convert("RGB")
        width = int(payload["width"])
        height = int(payload["height"])
        if width > 0 and height > 0:
            image = image.resize((width, height))
        generator = None
        if payload["seed"] is not None:
            gen_device = "cuda" if device == "cuda" else "cpu"
            generator = torch.Generator(device=gen_device).manual_seed(int(payload["seed"]))
        step_count = int(payload["num_inference_steps"])
        update_task(task_id, progress=0.2, message=f"Generating image (0/{step_count})")
        step_progress_kwargs = build_step_progress_kwargs(
            pipe=pipe,
            task_id=task_id,
            num_inference_steps=step_count,
            start_progress=0.2,
            end_progress=0.9,
            message="Generating image",
        )
        LOGGER.info(
            "image2image inference start task_id=%s model=%s steps=%s guidance=%s strength=%s callback_keys=%s",
            task_id,
            model_ref,
            payload["num_inference_steps"],
            payload["guidance_scale"],
            payload["strength"],
            ",".join(sorted(step_progress_kwargs.keys())) if step_progress_kwargs else "(none)",
        )
        gen_started = time.perf_counter()
        out = call_with_supported_kwargs(
            pipe,
            {
                "prompt": payload["prompt"],
                "negative_prompt": payload["negative_prompt"] or None,
                "image": image,
                "num_inference_steps": int(payload["num_inference_steps"]),
                "guidance_scale": float(payload["guidance_scale"]),
                "strength": float(payload["strength"]),
                "generator": generator,
                "output_type": "pil",
                "cross_attention_kwargs": {"scale": lora_scale} if len(lora_refs) == 1 else None,
                **step_progress_kwargs,
            },
        )
        LOGGER.info("image2image inference done task_id=%s elapsed_ms=%.1f", task_id, (time.perf_counter() - gen_started) * 1000)
        images: list[Any]
        raw_images = out.images
        if isinstance(raw_images, list) and raw_images and isinstance(raw_images[0], Image.Image):
            images = raw_images
            update_task(task_id, progress=0.96, message="Postprocessing image")
        else:
            update_task(task_id, progress=0.92, message="Decoding latents")
            decode_started = time.perf_counter()
            log_gpu_memory_stats("before_decode", task_id)
            try:
                with task_progress_heartbeat(
                    task_id,
                    start_progress=0.92,
                    end_progress=0.955,
                    message="Decoding latents",
                    interval_sec=0.5,
                    estimated_duration_sec=12.0,
                ):
                    images = decode_latents_to_pil_images(pipe, raw_images)
            except torch.OutOfMemoryError:
                LOGGER.error(
                    "image2image decode OOM task_id=%s. CPU fallback is disabled for diagnostics.",
                    task_id,
                    exc_info=True,
                )
                log_gpu_memory_stats("decode_oom", task_id)
                raise
            LOGGER.info("image2image decode done task_id=%s elapsed_ms=%.1f", task_id, (time.perf_counter() - decode_started) * 1000)
            log_gpu_memory_stats("after_decode", task_id)
            update_task(task_id, progress=0.96, message="Postprocessing image")
        result_image = images[0]
        update_task(task_id, progress=0.98, message="Saving png")
        output_name = f"image2image_{task_id}.png"
        output_path = resolve_path(settings["paths"]["outputs_dir"]) / output_name
        with task_progress_heartbeat(
            task_id,
            start_progress=0.98,
            end_progress=0.995,
            message="Saving png",
            interval_sec=0.4,
            estimated_duration_sec=4.0,
        ):
            result_image.save(str(output_path))
        update_task(
            task_id,
            status="completed",
            progress=1.0,
            message="Done",
            result={"image_file": output_name, "model": model_ref, "loras": lora_refs, "vae": vae_ref or None},
        )
        LOGGER.info("image2image done task_id=%s output=%s", task_id, str(output_path))
    except Exception as exc:
        trace = format_exception_trace()
        LOGGER.exception("image2image failed task_id=%s model=%s diagnostics=%s", task_id, model_ref, runtime_diagnostics())
        update_task(task_id, status="error", message="Generation failed", error=str(exc), error_trace=trace)
    finally:
        if image_path.exists():
            image_path.unlink(missing_ok=True)


def download_model_worker(task_id: str, repo_id: str, revision: Optional[str], target_dir: Optional[str]) -> None:
    settings = settings_store.get()
    ensure_runtime_dirs(settings)
    clean_target_dir = (target_dir or "").strip()
    base_dir_raw = clean_target_dir or settings["paths"]["models_dir"]
    models_dir = resolve_path(base_dir_raw)
    models_dir.mkdir(parents=True, exist_ok=True)
    token = settings["huggingface"].get("token") or None
    model_dir = models_dir / sanitize_repo_id(repo_id)
    LOGGER.info("download start task_id=%s repo=%s revision=%s target=%s", task_id, repo_id, revision or "main", str(model_dir))
    try:
        update_task(task_id, status="running", progress=0.0, message=f"Preparing download {repo_id}", downloaded_bytes=0, total_bytes=None)

        civitai_model_id = parse_civitai_model_id(repo_id)
        if civitai_model_id is not None:
            payload = civitai_request_json(f"models/{civitai_model_id}")
            primary_file = extract_civitai_primary_file(payload)
            if not primary_file:
                raise RuntimeError(f"No downloadable file found on CivitAI model {civitai_model_id}")
            download_url = str(primary_file.get("downloadUrl") or "").strip()
            if not download_url:
                raise RuntimeError(f"Missing download URL on CivitAI model {civitai_model_id}")
            model_name = str(payload.get("name") or f"civitai-{civitai_model_id}")
            file_name = sanitize_download_filename(str(primary_file.get("name") or ""), f"civitai-{civitai_model_id}.safetensors")
            model_dir.mkdir(parents=True, exist_ok=True)
            file_path = model_dir / file_name
            total_bytes_hint: Optional[int] = None
            size_kb_raw = primary_file.get("sizeKB")
            if isinstance(size_kb_raw, (int, float)):
                total_bytes_hint = int(float(size_kb_raw) * 1024)
            downloaded_bytes = stream_http_download(
                download_url,
                file_path,
                task_id,
                f"Downloading {repo_id}",
                total_bytes_hint=total_bytes_hint,
            )
            metadata = {
                "source": "civitai",
                "repo_id": repo_id,
                "model_id": civitai_model_id,
                "model_name": model_name,
                "model_url": f"https://civitai.com/models/{civitai_model_id}",
                "version_id": primary_file.get("version_id"),
                "version_name": primary_file.get("version_name"),
                "file_id": primary_file.get("id"),
                "file_name": file_name,
                "bytes": downloaded_bytes,
            }
            (model_dir / "civitai_model.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
        else:
            model_dir.mkdir(parents=True, exist_ok=True)
            api = HfApi(token=token)
            expected_total_bytes = get_remote_model_size_bytes(api, None, repo_id)
            initial_downloaded = directory_size_bytes(model_dir)
            update_task(
                task_id,
                progress=min(0.99, (initial_downloaded / expected_total_bytes)) if expected_total_bytes else 0.0,
                message=f"Downloading {repo_id}",
                downloaded_bytes=initial_downloaded,
                total_bytes=expected_total_bytes,
            )

            err_holder: Dict[str, Exception] = {}

            def run_hf_download() -> None:
                try:
                    snapshot_download(
                        repo_id=repo_id,
                        revision=revision,
                        local_dir=str(model_dir),
                        local_dir_use_symlinks=False,
                        token=token,
                        resume_download=True,
                    )
                except Exception as exc:
                    err_holder["error"] = exc

            worker = threading.Thread(target=run_hf_download, daemon=True)
            worker.start()
            last_reported_bytes = -1
            while worker.is_alive():
                downloaded_now = directory_size_bytes(model_dir)
                if downloaded_now != last_reported_bytes:
                    last_reported_bytes = downloaded_now
                    progress = min(0.95, (downloaded_now / (200 * 1024 * 1024)))
                    if expected_total_bytes and expected_total_bytes > 0:
                        progress = min(0.99, downloaded_now / expected_total_bytes)
                    update_task(
                        task_id,
                        progress=progress,
                        message=f"Downloading {repo_id}",
                        downloaded_bytes=downloaded_now,
                        total_bytes=expected_total_bytes,
                    )
                worker.join(timeout=1.0)
            worker.join(timeout=1.0)
            if err_holder.get("error"):
                raise err_holder["error"]
            downloaded_final = directory_size_bytes(model_dir)
            update_task(
                task_id,
                progress=min(0.99, downloaded_final / expected_total_bytes) if expected_total_bytes else 0.95,
                message=f"Downloading {repo_id}",
                downloaded_bytes=downloaded_final,
                total_bytes=expected_total_bytes or downloaded_final,
            )
        final_downloaded_bytes = directory_size_bytes(model_dir)
        current_task = get_task(task_id) or {}
        total_bytes_value = safe_int(current_task.get("total_bytes")) or final_downloaded_bytes
        update_task(
            task_id,
            status="completed",
            progress=1.0,
            message="Download complete",
            downloaded_bytes=final_downloaded_bytes,
            total_bytes=total_bytes_value,
            result={"repo_id": repo_id, "local_path": str(model_dir.resolve()), "base_dir": str(models_dir.resolve())},
        )
        LOGGER.info("download done task_id=%s repo=%s local_path=%s", task_id, repo_id, str(model_dir.resolve()))
    except Exception as exc:
        trace = format_exception_trace()
        LOGGER.exception("download failed task_id=%s repo=%s target=%s", task_id, repo_id, str(model_dir))
        update_task(task_id, status="error", message="Download failed", error=str(exc), error_trace=trace)


app = FastAPI(title="ROCm VideoGen")
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.on_event("startup")
def startup_preload_default_t2i() -> None:
    if not server_flag("preload_default_t2i_on_startup", True):
        LOGGER.info("startup preload skipped: preload_default_t2i_on_startup=false")
        return
    if bool(getattr(torch.version, "hip", None)) and not parse_bool_setting(
        os.environ.get("VIDEOGEN_ALLOW_ROCM_STARTUP_PRELOAD", "0"),
        default=False,
    ):
        LOGGER.warning(
            "startup preload skipped on ROCm for stability. "
            "Set VIDEOGEN_ALLOW_ROCM_STARTUP_PRELOAD=1 to force."
        )
        return
    if start_preload_default_t2i("startup"):
        LOGGER.info("startup preload scheduled: text-to-image")
    else:
        LOGGER.info("startup preload already running")


def read_last_lines(path: Path, line_count: int) -> list[str]:
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        return [line.rstrip("\n") for line in deque(handle, maxlen=line_count)]


def is_noisy_debug_path(path: str) -> bool:
    return any(path.startswith(prefix) for prefix in NOISY_DEBUG_ENDPOINT_PREFIXES)


@app.middleware("http")
async def request_log_middleware(request: Request, call_next: Any) -> Any:
    started = time.perf_counter()
    try:
        response = await call_next(request)
    except Exception:
        LOGGER.exception("request failed method=%s path=%s query=%s", request.method, request.url.path, request.url.query)
        raise
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    path = request.url.path
    should_log = response.status_code >= 400 or path in HIGH_VALUE_ENDPOINTS
    noisy_debug_path = is_noisy_debug_path(path)
    if should_log:
        LOGGER.info(
            "request method=%s path=%s query=%s status=%s elapsed_ms=%.1f",
            request.method,
            path,
            request.url.query,
            response.status_code,
            elapsed_ms,
        )
    elif noisy_debug_path:
        # Keep task polling logs compact at DEBUG to avoid heavy CPU/log I/O overhead.
        if elapsed_ms >= 300.0:
            LOGGER.debug(
                "request(slow) method=%s path=%s status=%s elapsed_ms=%.1f",
                request.method,
                path,
                response.status_code,
                elapsed_ms,
            )
    else:
        LOGGER.debug(
            "request method=%s path=%s query=%s status=%s elapsed_ms=%.1f",
            request.method,
            path,
            request.url.query,
            response.status_code,
            elapsed_ms,
        )
    return response


@app.get("/")
def root() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/system/info")
def system_info() -> Dict[str, Any]:
    return detect_runtime()


@app.get("/api/system/preload")
def system_preload_state() -> Dict[str, Any]:
    return get_preload_state()


@app.post("/api/models/preload-default-t2i")
def preload_default_t2i() -> Dict[str, Any]:
    started = start_preload_default_t2i("manual")
    state = get_preload_state()
    state["started"] = started
    return state


@app.get("/api/settings")
def get_settings() -> Dict[str, Any]:
    return settings_store.get()


@app.get("/api/logs/recent")
def recent_logs(lines: int = 200) -> Dict[str, Any]:
    settings = settings_store.get()
    line_limit = min(max(lines, 10), 2000)
    log_file = get_log_file_path(settings)
    if not log_file.exists():
        latest = latest_log_file(settings)
        if latest is None:
            raise HTTPException(status_code=404, detail=f"Log file not found: {log_file}")
        log_file = latest
    return {
        "log_file": str(log_file),
        "lines": read_last_lines(log_file, line_limit),
    }


@app.post("/api/cache/hf/clear")
def clear_hf_cache(req: ClearHfCacheRequest) -> Dict[str, Any]:
    candidates = sorted(gather_hf_cache_candidates(), key=lambda p: str(p).lower())
    removed: list[str] = []
    skipped: list[Dict[str, str]] = []
    failed: list[Dict[str, str]] = []
    for target in candidates:
        target_str = str(target)
        if not is_safe_cache_target(target):
            skipped.append({"path": target_str, "reason": "unsafe"})
            continue
        if not target.exists():
            skipped.append({"path": target_str, "reason": "not_found"})
            continue
        if req.dry_run:
            skipped.append({"path": target_str, "reason": "dry_run"})
            continue
        try:
            shutil.rmtree(target)
            removed.append(target_str)
        except Exception as exc:
            failed.append({"path": target_str, "error": str(exc)})
    LOGGER.info(
        "hf cache clear dry_run=%s candidates=%s removed=%s skipped=%s failed=%s",
        req.dry_run,
        len(candidates),
        len(removed),
        len(skipped),
        len(failed),
    )
    status = "ok" if not failed else "partial"
    return {
        "status": status,
        "dry_run": req.dry_run,
        "candidates": [str(path) for path in candidates],
        "removed_paths": removed,
        "skipped": skipped,
        "failed": failed,
    }


@app.put("/api/settings")
def update_settings(payload: Dict[str, Any]) -> Dict[str, Any]:
    updated = settings_store.update(payload)
    ensure_runtime_dirs(updated)
    setup_logger(updated)
    LOGGER.info(
        "settings updated models_dir=%s outputs_dir=%s tmp_dir=%s logs_dir=%s log_level=%s listen_port=%s rocm_aotriton_experimental=%s require_gpu=%s allow_cpu_fallback=%s preload_default_t2i_on_startup=%s t2v_backend=%s t2v_npu_runner=%s t2v_npu_model_dir=%s",
        updated["paths"].get("models_dir"),
        updated["paths"].get("outputs_dir"),
        updated["paths"].get("tmp_dir"),
        updated["paths"].get("logs_dir"),
        updated.get("logging", {}).get("level"),
        updated.get("server", {}).get("listen_port"),
        updated.get("server", {}).get("rocm_aotriton_experimental"),
        updated.get("server", {}).get("require_gpu"),
        updated.get("server", {}).get("allow_cpu_fallback"),
        updated.get("server", {}).get("preload_default_t2i_on_startup"),
        updated.get("server", {}).get("t2v_backend"),
        updated.get("server", {}).get("t2v_npu_runner"),
        updated.get("server", {}).get("t2v_npu_model_dir"),
    )
    return updated


@app.get("/api/models/local")
def list_local_models(dir: str = "") -> Dict[str, Any]:
    settings = settings_store.get()
    models_dir = resolve_path(dir.strip() or settings["paths"]["models_dir"])
    items = []
    if models_dir.exists():
        for child in sorted(models_dir.iterdir()):
            if child.is_dir():
                meta = detect_local_model_meta(child)
                preview_rel = find_local_preview_relpath(child, models_dir)
                preview_url = None
                if preview_rel:
                    preview_url = (
                        f"/api/models/preview?rel={quote(preview_rel, safe='/')}"
                        f"&base_dir={quote(str(models_dir), safe='/:\\\\')}"
                    )
                items.append(
                    {
                        "name": child.name,
                        "repo_hint": desanitize_repo_id(child.name),
                        "path": str(child.resolve()),
                        "can_delete": is_deletable_model_dir(child),
                        "preview_url": preview_url,
                        "class_name": meta.class_name,
                        "base_model": meta.base_model,
                        "compatible_tasks": meta.compatible_tasks,
                        "is_lora": meta.is_lora,
                        "is_vae": meta.is_vae,
                    }
                )
            elif is_single_file_model(child):
                compatible_tasks = single_file_model_compatible_tasks(child)
                items.append(
                    {
                        "name": child.name,
                        "repo_hint": child.stem,
                        "path": str(child.resolve()),
                        "can_delete": False,
                        "preview_url": None,
                        "class_name": "SingleFileCheckpoint",
                        "base_model": single_file_base_model_label(child),
                        "compatible_tasks": compatible_tasks,
                        "is_lora": False,
                        "is_vae": False,
                    }
                )
    return {"items": items, "base_dir": str(models_dir)}


@app.post("/api/models/local/delete")
def delete_local_model(req: DeleteLocalModelRequest) -> Dict[str, Any]:
    settings = settings_store.get()
    base_dir = resolve_path((req.base_dir or "").strip() or settings["paths"]["models_dir"])
    model_name = req.model_name.strip()
    if not model_name:
        raise HTTPException(status_code=400, detail="model_name is required")
    if Path(model_name).name != model_name or "/" in model_name or "\\" in model_name:
        raise HTTPException(status_code=400, detail="Invalid model_name")

    target = (base_dir / model_name).resolve()
    if not safe_in_directory(target, base_dir):
        raise HTTPException(status_code=400, detail="Invalid target path")
    if target.parent.resolve() != base_dir.resolve():
        raise HTTPException(status_code=400, detail="Only direct child directories can be deleted")
    if not target.exists() or not target.is_dir():
        raise HTTPException(status_code=404, detail="Model directory not found")
    if not is_deletable_model_dir(target):
        raise HTTPException(status_code=400, detail="Target directory is not recognized as a downloadable model")

    shutil.rmtree(target)
    LOGGER.info("local model deleted base_dir=%s name=%s path=%s", str(base_dir), model_name, str(target))
    return {"status": "ok", "deleted_path": str(target), "model_name": model_name}


@app.get("/api/models/preview")
def get_local_model_preview(rel: str, base_dir: str = "") -> FileResponse:
    settings = settings_store.get()
    models_dir = resolve_path(base_dir.strip() or settings["paths"]["models_dir"])
    requested = (models_dir / rel).resolve()
    if not safe_in_directory(requested, models_dir):
        raise HTTPException(status_code=400, detail="Invalid preview path")
    if requested.suffix.lower() not in (".png", ".jpg", ".jpeg", ".webp"):
        raise HTTPException(status_code=400, detail="Unsupported preview type")
    if not requested.exists():
        raise HTTPException(status_code=404, detail="Preview not found")
    media_type = "image/webp" if requested.suffix.lower() == ".webp" else "image/jpeg"
    if requested.suffix.lower() == ".png":
        media_type = "image/png"
    return FileResponse(requested, media_type=media_type)


@app.get("/api/models/catalog")
def model_catalog(task: Literal["text-to-image", "image-to-image", "text-to-video", "image-to-video"], limit: int = 30) -> Dict[str, Any]:
    settings = settings_store.get()
    models_dir = resolve_path(settings["paths"]["models_dir"])
    models_dir.mkdir(parents=True, exist_ok=True)

    local_items = []
    for child in sorted(models_dir.iterdir()):
        if child.is_dir():
            if not is_local_model_compatible(task, child):
                LOGGER.info("catalog skip incompatible task=%s path=%s", task, str(child))
                continue
            repo_hint = desanitize_repo_id(child.name)
            preview_rel = find_local_preview_relpath(child, models_dir)
            preview_url = None
            if preview_rel:
                preview_url = f"/api/models/preview?rel={quote(preview_rel, safe='/')}"
            local_items.append(
                {
                    "source": "local",
                    "label": f"[local] {repo_hint}",
                    "value": str(child.resolve()),
                    "id": repo_hint,
                    "size_bytes": None,
                    "preview_url": preview_url,
                    "model_url": f"https://huggingface.co/{quote(repo_hint, safe='/')}",
                }
            )
            continue
        if is_single_file_model(child):
            compatible_tasks = single_file_model_compatible_tasks(child)
            if task not in compatible_tasks:
                continue
            size_bytes = None
            with contextlib.suppress(OSError):
                size_bytes = int(child.stat().st_size)
            local_items.append(
                {
                    "source": "local",
                    "label": f"[local] {child.stem}",
                    "value": str(child.resolve()),
                    "id": child.stem,
                    "size_bytes": size_bytes,
                    "preview_url": None,
                    "model_url": None,
                }
            )

    default_model = get_default_model_for_task(task, settings)
    return {"items": local_items, "default_model": default_model}


@app.get("/api/models/search")
def search_models(
    task: Literal["text-to-image", "image-to-image", "text-to-video", "image-to-video"],
    query: str = "",
    limit: int = 30,
    source: Literal["all", "huggingface", "civitai"] = "all",
    base_model: str = "",
) -> Dict[str, Any]:
    settings = settings_store.get()
    token = settings["huggingface"].get("token") or None
    capped_limit = min(max(limit, 1), 50)
    include_hf = source in ("all", "huggingface")
    include_civitai = source in ("all", "civitai") and task in ("text-to-image", "image-to-image")

    hf_limit = 0
    civitai_limit = 0
    if include_hf and include_civitai:
        civitai_limit = max(1, capped_limit // 2)
        hf_limit = max(1, capped_limit - civitai_limit)
    elif include_hf:
        hf_limit = capped_limit
    elif include_civitai:
        civitai_limit = capped_limit

    hf_results = search_hf_models(task=task, query=query, limit=hf_limit, token=token) if hf_limit > 0 else []
    civitai_results = search_civitai_models(task=task, query=query, limit=civitai_limit) if civitai_limit > 0 else []
    results: list[Dict[str, Any]] = []
    max_len = max(len(hf_results), len(civitai_results))
    for index in range(max_len):
        if index < len(hf_results):
            results.append(hf_results[index])
        if index < len(civitai_results):
            results.append(civitai_results[index])
        if len(results) >= capped_limit:
            break
    normalized_base_model = normalize_base_model_filter(base_model)
    if normalized_base_model:
        results = [
            item
            for item in results
            if matches_base_model_filter(
                str(item.get("base_model") or infer_base_model_label(item.get("id"), item.get("pipeline_tag"), item.get("type"))),
                normalized_base_model,
            )
        ]
    for item in results:
        if not item.get("base_model"):
            item["base_model"] = infer_base_model_label(item.get("id"), item.get("pipeline_tag"), item.get("type"))
    return {"items": results}


@app.get("/api/models/loras/catalog")
def lora_catalog(
    task: Literal["text-to-image", "image-to-image", "text-to-video", "image-to-video"],
    model_ref: str = "",
    limit: int = 200,
) -> Dict[str, Any]:
    settings = settings_store.get()
    models_dir = resolve_path(settings["paths"]["models_dir"])
    models_dir.mkdir(parents=True, exist_ok=True)

    effective_model = model_ref.strip() or get_default_model_for_task(task, settings)
    capped_limit = min(max(limit, 1), 1000)
    items: list[Dict[str, Any]] = []
    for child in sorted(models_dir.iterdir()):
        if not child.is_dir():
            continue
        if not is_local_lora_dir(child):
            continue
        adapter_config = parse_lora_adapter_config(child)
        base_model_hint = str(adapter_config.get("base_model_name_or_path") or "").strip()
        repo_hint = desanitize_repo_id(child.name)
        if task in ("text-to-video", "image-to-video") and not base_model_hint:
            # Video tasks are strict: when adapter metadata is missing, skip unknown LoRA
            # to avoid incompatible adapters crashing inference (e.g. SDXL LoRA on T2V 3D UNet).
            inferred_lora_base = infer_base_model_label(repo_hint)
            inferred_model_base = infer_base_model_label(effective_model)
            if inferred_lora_base != inferred_model_base:
                continue
        if not lora_matches_model(base_model_hint, effective_model):
            continue
        preview_rel = find_local_preview_relpath(child, models_dir)
        preview_url = f"/api/models/preview?rel={quote(preview_rel, safe='/')}" if preview_rel else None
        items.append(
            {
                "source": "local",
                "label": f"[lora] {repo_hint}",
                "value": str(child.resolve()),
                "id": repo_hint,
                "base_model": base_model_hint or None,
                "size_bytes": local_lora_size_bytes(child),
                "preview_url": preview_url,
                "model_url": f"https://huggingface.co/{quote(repo_hint, safe='/')}",
            }
        )
        if len(items) >= capped_limit:
            break
    return {"items": items, "model_ref": effective_model}


@app.get("/api/models/vaes/catalog")
def vae_catalog(limit: int = 200) -> Dict[str, Any]:
    settings = settings_store.get()
    models_dir = resolve_path(settings["paths"]["models_dir"])
    models_dir.mkdir(parents=True, exist_ok=True)
    capped_limit = min(max(limit, 1), 1000)
    items: list[Dict[str, Any]] = []
    for child in sorted(models_dir.iterdir()):
        if not child.is_dir():
            continue
        if not is_local_vae_dir(child):
            continue
        repo_hint = desanitize_repo_id(child.name)
        preview_rel = find_local_preview_relpath(child, models_dir)
        preview_url = f"/api/models/preview?rel={quote(preview_rel, safe='/')}" if preview_rel else None
        items.append(
            {
                "source": "local",
                "label": f"[vae] {repo_hint}",
                "value": str(child.resolve()),
                "id": repo_hint,
                "size_bytes": None,
                "preview_url": preview_url,
                "model_url": f"https://huggingface.co/{quote(repo_hint, safe='/')}",
            }
        )
        if len(items) >= capped_limit:
            break
    return {"items": items}


@app.post("/api/models/download")
def download_model(req: DownloadRequest) -> Dict[str, str]:
    task_id = create_task("download", "Download queued")
    LOGGER.info("download requested task_id=%s repo=%s revision=%s target_dir=%s", task_id, req.repo_id, req.revision, req.target_dir)
    thread = threading.Thread(
        target=download_model_worker,
        args=(task_id, req.repo_id, req.revision, req.target_dir),
        daemon=True,
    )
    thread.start()
    return {"task_id": task_id}


@app.post("/api/generate/text2image")
def generate_text2image(req: Text2ImageRequest) -> Dict[str, str]:
    try:
        assert_generation_runtime_ready()
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=f"Diffusers runtime is not available: {exc}") from exc
    task_id = create_task("text2image", "Generation queued")
    lora_refs = collect_lora_refs(req.lora_id, req.lora_ids)
    LOGGER.info(
        "text2image requested task_id=%s model=%s loras=%s lora_scale=%s vae=%s prompt_len=%s negative_len=%s steps=%s guidance=%s size=%sx%s seed=%s",
        task_id,
        req.model_id or "(default)",
        ",".join(lora_refs) if lora_refs else "(none)",
        req.lora_scale,
        (req.vae_id or "").strip() or "(none)",
        len(req.prompt or ""),
        len(req.negative_prompt or ""),
        req.num_inference_steps,
        req.guidance_scale,
        req.width,
        req.height,
        req.seed,
    )
    thread = threading.Thread(target=text2image_worker, args=(task_id, req), daemon=True)
    thread.start()
    return {"task_id": task_id}


@app.post("/api/generate/image2image")
async def generate_image2image(
    image: UploadFile = File(...),
    prompt: str = Form(...),
    negative_prompt: str = Form(""),
    model_id: str = Form(""),
    lora_id: str = Form(""),
    lora_ids: list[str] = Form(default=[]),
    lora_scale: float = Form(1.0),
    vae_id: str = Form(""),
    num_inference_steps: int = Form(30),
    guidance_scale: float = Form(7.5),
    strength: float = Form(0.8),
    width: int = Form(512),
    height: int = Form(512),
    seed: Optional[int] = Form(None),
) -> Dict[str, str]:
    try:
        assert_generation_runtime_ready()
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=f"Diffusers runtime is not available: {exc}") from exc
    settings = settings_store.get()
    tmp_dir = resolve_path(settings["paths"]["tmp_dir"])
    tmp_dir.mkdir(parents=True, exist_ok=True)
    suffix = Path(image.filename or "input.png").suffix or ".png"
    tmp_path = tmp_dir / f"{uuid.uuid4()}{suffix}"
    with tmp_path.open("wb") as out_file:
        shutil.copyfileobj(image.file, out_file)
    lora_refs = collect_lora_refs(lora_id, lora_ids)
    payload = {
        "image_path": str(tmp_path),
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "model_id": model_id.strip(),
        "lora_id": lora_id.strip(),
        "lora_ids": lora_refs,
        "lora_scale": lora_scale,
        "vae_id": vae_id.strip(),
        "num_inference_steps": num_inference_steps,
        "guidance_scale": guidance_scale,
        "strength": strength,
        "width": width,
        "height": height,
        "seed": seed,
    }
    task_id = create_task("image2image", "Generation queued")
    LOGGER.info(
        "image2image requested task_id=%s model=%s loras=%s lora_scale=%s vae=%s file=%s steps=%s guidance=%s strength=%s size=%sx%s seed=%s",
        task_id,
        model_id.strip() or "(default)",
        ",".join(lora_refs) if lora_refs else "(none)",
        lora_scale,
        vae_id.strip() or "(none)",
        image.filename,
        num_inference_steps,
        guidance_scale,
        strength,
        width,
        height,
        seed,
    )
    thread = threading.Thread(target=image2image_worker, args=(task_id, payload), daemon=True)
    thread.start()
    return {"task_id": task_id}


@app.post("/api/generate/text2video")
def generate_text2video(req: Text2VideoRequest) -> Dict[str, str]:
    settings = settings_store.get()
    backend = resolve_text2video_backend(req.backend, settings)
    lora_refs = collect_lora_refs(req.lora_id, req.lora_ids)
    if backend == "npu":
        runner_raw = str(settings.get("server", {}).get("t2v_npu_runner", "")).strip()
        if not runner_raw:
            raise HTTPException(
                status_code=400,
                detail=(
                    "NPU backend requires server.t2v_npu_runner. "
                    "Set it in Settings (T2V NPU Runner)."
                ),
            )
        runner_path = Path(runner_raw).expanduser()
        if not runner_path.is_absolute():
            runner_path = (BASE_DIR / runner_path).resolve()
        if not runner_path.exists():
            raise HTTPException(
                status_code=400,
                detail=f"NPU runner not found: {runner_path}",
            )
        if lora_refs:
            raise HTTPException(
                status_code=400,
                detail="NPU backend for text-to-video does not support LoRA yet.",
            )
    if backend == "cuda":
        try:
            assert_generation_runtime_ready()
        except RuntimeError as exc:
            raise HTTPException(status_code=500, detail=f"Diffusers runtime is not available: {exc}") from exc
    task_id = create_task("text2video", "Generation queued")
    LOGGER.info(
        "text2video requested task_id=%s model=%s backend=%s requested_backend=%s loras=%s lora_scale=%s prompt_len=%s negative_len=%s steps=%s frames=%s guidance=%s fps=%s seed=%s",
        task_id,
        req.model_id or "(default)",
        backend,
        req.backend,
        ",".join(lora_refs) if lora_refs else "(none)",
        req.lora_scale,
        len(req.prompt or ""),
        len(req.negative_prompt or ""),
        req.num_inference_steps,
        req.num_frames,
        req.guidance_scale,
        req.fps,
        req.seed,
    )
    thread = threading.Thread(target=text2video_worker, args=(task_id, req), daemon=True)
    thread.start()
    return {"task_id": task_id}


@app.post("/api/generate/image2video")
async def generate_image2video(
    image: UploadFile = File(...),
    prompt: str = Form(...),
    negative_prompt: str = Form(""),
    model_id: str = Form(""),
    lora_id: str = Form(""),
    lora_ids: list[str] = Form(default=[]),
    lora_scale: float = Form(1.0),
    num_inference_steps: int = Form(30),
    num_frames: int = Form(16),
    guidance_scale: float = Form(9.0),
    fps: int = Form(8),
    width: int = Form(512),
    height: int = Form(512),
    seed: Optional[int] = Form(None),
) -> Dict[str, str]:
    try:
        assert_generation_runtime_ready()
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=f"Diffusers runtime is not available: {exc}") from exc
    settings = settings_store.get()
    tmp_dir = resolve_path(settings["paths"]["tmp_dir"])
    tmp_dir.mkdir(parents=True, exist_ok=True)
    suffix = Path(image.filename or "input.png").suffix or ".png"
    tmp_path = tmp_dir / f"{uuid.uuid4()}{suffix}"
    with tmp_path.open("wb") as out_file:
        shutil.copyfileobj(image.file, out_file)
    lora_refs = collect_lora_refs(lora_id, lora_ids)
    payload = {
        "image_path": str(tmp_path),
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "model_id": model_id.strip(),
        "lora_id": lora_id.strip(),
        "lora_ids": lora_refs,
        "lora_scale": lora_scale,
        "num_inference_steps": num_inference_steps,
        "num_frames": num_frames,
        "guidance_scale": guidance_scale,
        "fps": fps,
        "width": width,
        "height": height,
        "seed": seed,
    }
    task_id = create_task("image2video", "Generation queued")
    LOGGER.info(
        "image2video requested task_id=%s model=%s loras=%s lora_scale=%s file=%s steps=%s frames=%s guidance=%s fps=%s size=%sx%s seed=%s",
        task_id,
        model_id.strip() or "(default)",
        ",".join(lora_refs) if lora_refs else "(none)",
        lora_scale,
        image.filename,
        num_inference_steps,
        num_frames,
        guidance_scale,
        fps,
        width,
        height,
        seed,
    )
    thread = threading.Thread(target=image2video_worker, args=(task_id, payload), daemon=True)
    thread.start()
    return {"task_id": task_id}


@app.get("/api/tasks/{task_id}")
def task_status(task_id: str) -> Dict[str, Any]:
    task = get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    return task


@app.get("/api/outputs")
def list_outputs(limit: int = 200) -> Dict[str, Any]:
    settings = settings_store.get()
    outputs_dir = resolve_path(settings["paths"]["outputs_dir"])
    outputs_dir.mkdir(parents=True, exist_ok=True)
    capped_limit = min(max(limit, 1), 1000)
    entries: list[Dict[str, Any]] = []
    for child in outputs_dir.iterdir():
        if not child.is_file():
            continue
        kind = detect_output_kind(child)
        stat = child.stat()
        entries.append(
            {
                "name": child.name,
                "path": str(child.resolve()),
                "kind": kind,
                "size_bytes": int(stat.st_size),
                "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(stat.st_mtime)),
                "view_url": output_view_url(child.name, kind),
                "_sort_ts": float(stat.st_mtime),
            }
        )
    entries.sort(key=lambda item: float(item.get("_sort_ts") or 0.0), reverse=True)
    for item in entries:
        item.pop("_sort_ts", None)
    return {"base_dir": str(outputs_dir), "items": entries[:capped_limit]}


@app.post("/api/outputs/delete")
def delete_output(req: DeleteOutputRequest) -> Dict[str, Any]:
    settings = settings_store.get()
    outputs_dir = resolve_path(settings["paths"]["outputs_dir"])
    outputs_dir.mkdir(parents=True, exist_ok=True)
    file_name = req.file_name.strip()
    if not file_name:
        raise HTTPException(status_code=400, detail="file_name is required")
    if Path(file_name).name != file_name or "/" in file_name or "\\" in file_name:
        raise HTTPException(status_code=400, detail="Invalid file_name")
    target = (outputs_dir / file_name).resolve()
    if not safe_in_directory(target, outputs_dir):
        raise HTTPException(status_code=400, detail="Invalid target path")
    if not target.exists():
        raise HTTPException(status_code=404, detail="Output file not found")
    if not target.is_file():
        raise HTTPException(status_code=400, detail="Target is not a file")
    target.unlink()
    LOGGER.info("output deleted path=%s", str(target))
    return {"status": "ok", "file_name": file_name, "deleted_path": str(target)}


@app.get("/api/videos/{file_name}")
def get_video(file_name: str) -> FileResponse:
    settings = settings_store.get()
    outputs_dir = resolve_path(settings["paths"]["outputs_dir"])
    requested = (outputs_dir / file_name).resolve()
    if not safe_in_directory(requested, outputs_dir):
        raise HTTPException(status_code=400, detail="Invalid path")
    suffix = requested.suffix.lower()
    if suffix not in OUTPUT_VIDEO_EXTENSIONS:
        raise HTTPException(status_code=400, detail="Unsupported video type")
    if not requested.exists():
        raise HTTPException(status_code=404, detail="File not found")
    media_type = {
        ".mp4": "video/mp4",
        ".mov": "video/quicktime",
        ".webm": "video/webm",
        ".mkv": "video/x-matroska",
    }.get(suffix, "application/octet-stream")
    return FileResponse(requested, media_type=media_type)


@app.get("/api/images/{file_name}")
def get_image(file_name: str) -> FileResponse:
    settings = settings_store.get()
    outputs_dir = resolve_path(settings["paths"]["outputs_dir"])
    requested = (outputs_dir / file_name).resolve()
    if not safe_in_directory(requested, outputs_dir):
        raise HTTPException(status_code=400, detail="Invalid path")
    if requested.suffix.lower() not in OUTPUT_IMAGE_EXTENSIONS:
        raise HTTPException(status_code=400, detail="Unsupported image type")
    if not requested.exists():
        raise HTTPException(status_code=404, detail="File not found")
    media_type = "image/webp" if requested.suffix.lower() == ".webp" else "image/jpeg"
    if requested.suffix.lower() == ".png":
        media_type = "image/png"
    return FileResponse(requested, media_type=media_type)
