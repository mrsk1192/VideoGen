# Full File Snapshot


===== FILE: main.py =====

import copy
import contextlib
import dataclasses
import hashlib
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
from typing import Any, Callable, Dict, Literal, Optional
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode
from urllib.request import Request as UrlRequest, urlopen

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from huggingface_hub import HfApi, hf_hub_download, snapshot_download
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
SEARCH_API_CACHE: Dict[str, Dict[str, Any]] = {}
SEARCH_API_CACHE_LOCK = threading.Lock()
SEARCH_API_CACHE_TTL_SEC = 60 * 5
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
    "/api/models/search2",
    "/api/models/detail",
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


def normalized_task(task: str) -> str:
    raw = str(task or "").strip().lower()
    if raw in ("text-to-image", "image-to-image", "text-to-video", "image-to-video"):
        return raw
    return "text-to-image"


def normalized_source(source: str) -> str:
    raw = str(source or "").strip().lower()
    if raw in ("all", "huggingface", "civitai"):
        return raw
    return "all"


def normalized_sort(sort: str) -> str:
    raw = str(sort or "").strip().lower()
    if raw in ("popularity", "downloads", "likes", "updated", "created"):
        return raw
    return "downloads"


def normalized_nsfw(nsfw: str) -> str:
    raw = str(nsfw or "").strip().lower()
    if raw in ("include", "exclude"):
        return raw
    return "exclude"


def normalized_model_kind(model_kind: str) -> str:
    raw = str(model_kind or "").strip().lower()
    if raw in ("checkpoint", "lora", "vae", "controlnet", "embedding", "upscaler"):
        return raw
    return ""


def cache_key(prefix: str, payload: Dict[str, Any]) -> str:
    body = json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(body.encode("utf-8")).hexdigest()
    return f"{prefix}:{digest}"


def cache_get_or_set(prefix: str, payload: Dict[str, Any], loader: Callable[[], Dict[str, Any]]) -> Dict[str, Any]:
    key = cache_key(prefix, payload)
    now = time.time()
    with SEARCH_API_CACHE_LOCK:
        cached = SEARCH_API_CACHE.get(key)
        if cached and (now - float(cached.get("ts") or 0.0)) < SEARCH_API_CACHE_TTL_SEC:
            value = cached.get("value")
            if isinstance(value, dict):
                return copy.deepcopy(value)
    loaded = loader()
    with SEARCH_API_CACHE_LOCK:
        SEARCH_API_CACHE[key] = {"ts": now, "value": copy.deepcopy(loaded)}
    return loaded


def normalize_model_lookup_key(value: str) -> str:
    return str(value or "").strip().lower().replace("\\", "/")


def collect_installed_model_id_set(models_dir: Path) -> set[str]:
    installed: set[str] = set()
    if not models_dir.exists():
        return installed
    for child in models_dir.iterdir():
        if child.is_dir():
            repo_hint = desanitize_repo_id(child.name)
            installed.add(normalize_model_lookup_key(repo_hint))
            installed.add(normalize_model_lookup_key(child.name))
            civitai_meta = child / "civitai_model.json"
            if civitai_meta.exists():
                with contextlib.suppress(Exception):
                    payload = json.loads(civitai_meta.read_text(encoding="utf-8"))
                    model_id = safe_int(payload.get("model_id"))
                    if model_id:
                        installed.add(normalize_model_lookup_key(f"civitai/{model_id}"))
        elif is_single_file_model(child):
            installed.add(normalize_model_lookup_key(child.stem))
    return installed


def hf_sort_to_api(sort: str) -> str:
    normalized = normalized_sort(sort)
    if normalized in ("popularity", "downloads"):
        return "downloads"
    if normalized == "likes":
        return "likes"
    if normalized == "updated":
        return "lastModified"
    if normalized == "created":
        return "createdAt"
    return "downloads"


def hf_model_matches_kind(model: Any, model_kind: str) -> bool:
    normalized_kind = normalized_model_kind(model_kind)
    if not normalized_kind:
        return True
    tags = [str(tag).lower() for tag in (getattr(model, "tags", None) or [])]
    model_id = str(getattr(model, "id", "")).lower()
    joined = " ".join(tags + [model_id])
    if normalized_kind == "checkpoint":
        return not any(token in joined for token in ("lora", "vae", "controlnet", "embedding", "upscaler"))
    return normalized_kind in joined


def civitai_task_model_types(task: str, model_kind: str = "") -> list[str]:
    normalized_kind = normalized_model_kind(model_kind)
    kind_map = {
        "checkpoint": ["Checkpoint"],
        "lora": ["LORA"],
        "vae": ["VAE"],
        "controlnet": ["Controlnet"],
        "embedding": ["TextualInversion"],
        "upscaler": ["Upscaler"],
    }
    if normalized_kind:
        return kind_map.get(normalized_kind, [])
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


def parse_civitai_item(item: Dict[str, Any], task: str) -> Optional[Dict[str, Any]]:
    model_id = safe_int(item.get("id"))
    if not model_id:
        return None
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
        if isinstance(files, list):
            for candidate_file in files:
                if not isinstance(candidate_file, dict):
                    continue
                if str(candidate_file.get("downloadUrl") or "").strip():
                    has_download_file = True
                    size_kb = candidate_file.get("sizeKB")
                    if isinstance(size_kb, (int, float)) and size_kb > 0:
                        size_bytes = int(float(size_kb) * 1024)
                    break
    if not base_model_raw:
        base_model_raw = str(item.get("baseModel") or "").strip()
    base_model = infer_base_model_label(base_model_raw, model_name, model_type, task)
    result_id = f"civitai/{model_id}"
    return {
        "id": result_id,
        "name": model_name,
        "title": model_name,
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
        "civitai_model_id": model_id,
    }


def search_hf_models_v2(
    task: Literal["text-to-image", "image-to-image", "text-to-video", "image-to-video"],
    query: str,
    limit: int,
    token: Optional[str],
    page: int = 1,
    sort: str = "downloads",
    model_kind: str = "",
) -> Dict[str, Any]:
    capped_limit = min(max(limit, 1), 50)
    current_page = max(int(page), 1)
    normalized_query = query.strip()
    normalized_kind = normalized_model_kind(model_kind)
    normalized_sort_value = normalized_sort(sort)
    cache_payload = {
        "task": task,
        "query": normalized_query,
        "limit": capped_limit,
        "page": current_page,
        "sort": normalized_sort_value,
        "model_kind": normalized_kind,
    }

    def loader() -> Dict[str, Any]:
        api = HfApi(token=token)
        fetch_limit = min(300, max(capped_limit * current_page * 3, capped_limit * 3))
        list_kwargs: Dict[str, Any] = {
            "sort": hf_sort_to_api(normalized_sort_value),
            "direction": -1,
            "limit": fetch_limit,
            "full": True,
            "cardData": True,
        }
        if normalized_query:
            list_kwargs["search"] = normalized_query
        else:
            list_kwargs["filter"] = task

        try:
            found = list(api.list_models(**list_kwargs))
        except Exception:
            LOGGER.warning("hf search2 list_models failed with sort=%s. fallback=downloads", list_kwargs.get("sort"), exc_info=True)
            list_kwargs["sort"] = "downloads"
            found = list(api.list_models(**list_kwargs))
        filtered: list[Dict[str, Any]] = []
        for model in found:
            pipeline_tag = getattr(model, "pipeline_tag", None)
            tags = getattr(model, "tags", []) or []
            matches_task = pipeline_tag == task or task in tags
            if not matches_task:
                continue
            if not hf_model_matches_kind(model, normalized_kind):
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
            filtered.append(
                {
                    "id": model_id,
                    "name": model_id,
                    "title": model_id,
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
                    "type": str(pipeline_tag or ""),
                }
            )
        start = (current_page - 1) * capped_limit
        page_items = filtered[start : start + capped_limit]
        has_next = len(filtered) > start + capped_limit
        return {"items": page_items, "has_next": has_next, "total_known": len(filtered)}

    return cache_get_or_set("hf-search2", cache_payload, loader)


def civitai_sort_to_api(sort: str) -> str:
    normalized = normalized_sort(sort)
    if normalized in ("popularity", "downloads"):
        return "Most Downloaded"
    if normalized == "likes":
        return "Most Liked"
    if normalized in ("updated", "created"):
        return "Newest"
    return "Most Downloaded"


def search_civitai_models_v2(
    task: Literal["text-to-image", "image-to-image", "text-to-video", "image-to-video"],
    query: str,
    limit: int,
    page: int = 1,
    sort: str = "downloads",
    nsfw: str = "exclude",
    model_kind: str = "",
) -> Dict[str, Any]:
    model_types = civitai_task_model_types(task, model_kind=model_kind)
    if not model_types:
        return {"items": [], "has_next": False}
    capped_limit = min(max(limit, 1), 100)
    current_page = max(int(page), 1)
    normalized_query = query.strip()
    normalized_sort_value = normalized_sort(sort)
    normalized_nsfw_value = normalized_nsfw(nsfw)
    normalized_kind = normalized_model_kind(model_kind)
    cache_payload = {
        "task": task,
        "query": normalized_query,
        "limit": capped_limit,
        "page": current_page,
        "sort": normalized_sort_value,
        "nsfw": normalized_nsfw_value,
        "model_kind": normalized_kind,
    }

    def loader() -> Dict[str, Any]:
        params: Dict[str, Any] = {
            "limit": capped_limit,
            "page": current_page,
            "nsfw": "true" if normalized_nsfw_value == "include" else "false",
            "sort": civitai_sort_to_api(normalized_sort_value),
            "period": "AllTime",
        }
        if normalized_query:
            params["query"] = normalized_query
        for model_type in model_types:
            params.setdefault("types", [])
            params["types"].append(model_type)
        try:
            payload = civitai_request_json("models", params)
        except (HTTPError, URLError, TimeoutError, json.JSONDecodeError, UnicodeDecodeError):
            LOGGER.warning("civitai search2 failed task=%s query=%s page=%s", task, query, page, exc_info=True)
            return {"items": [], "has_next": False}
        items = payload.get("items", []) if isinstance(payload, dict) else []
        results: list[Dict[str, Any]] = []
        for item in items:
            if not isinstance(item, dict):
                continue
            parsed = parse_civitai_item(item, task)
            if parsed:
                results.append(parsed)
        metadata = payload.get("metadata", {}) if isinstance(payload, dict) and isinstance(payload.get("metadata"), dict) else {}
        has_next = bool(metadata.get("nextPage"))
        return {"items": results[:capped_limit], "has_next": has_next}

    return cache_get_or_set("civitai-search2", cache_payload, loader)


def search_hf_models(
    task: Literal["text-to-image", "image-to-image", "text-to-video", "image-to-video"],
    query: str,
    limit: int,
    token: Optional[str],
) -> list[Dict[str, Any]]:
    result = search_hf_models_v2(task=task, query=query, limit=limit, token=token, page=1, sort="downloads", model_kind="")
    return list(result.get("items") or [])


def search_civitai_models(task: Literal["text-to-image", "image-to-image", "text-to-video", "image-to-video"], query: str, limit: int) -> list[Dict[str, Any]]:
    result = search_civitai_models_v2(task=task, query=query, limit=limit, page=1, sort="downloads", nsfw="exclude", model_kind="")
    return list(result.get("items") or [])


def collect_hf_preview_urls(repo_id: str, siblings: Any) -> list[str]:
    urls: list[str] = []
    for sibling in siblings or []:
        name = str(getattr(sibling, "rfilename", "") or "")
        if not name:
            continue
        lowered = name.lower()
        if lowered.endswith((".png", ".jpg", ".jpeg", ".webp")) and ("preview" in lowered or "thumb" in lowered):
            urls.append(f"https://huggingface.co/{quote(repo_id, safe='/')}/resolve/main/{quote(name, safe='/')}")
    if not urls:
        for sibling in siblings or []:
            name = str(getattr(sibling, "rfilename", "") or "")
            lowered = name.lower()
            if lowered.endswith((".png", ".jpg", ".jpeg", ".webp")):
                urls.append(f"https://huggingface.co/{quote(repo_id, safe='/')}/resolve/main/{quote(name, safe='/')}")
                if len(urls) >= 8:
                    break
    unique: list[str] = []
    seen: set[str] = set()
    for url in urls:
        if url in seen:
            continue
        seen.add(url)
        unique.append(url)
    return unique[:8]


def hf_repo_ref_versions(api: HfApi, repo_id: str, files: list[Dict[str, Any]]) -> list[Dict[str, Any]]:
    versions: list[Dict[str, Any]] = [{"id": "main", "name": "main", "createdAt": None, "files": files}]
    try:
        refs = api.list_repo_refs(repo_id=repo_id, repo_type="model")
    except Exception:
        return versions

    def version_files(revision: str) -> list[Dict[str, Any]]:
        output: list[Dict[str, Any]] = []
        for item in files:
            file_name = str(item.get("name") or "")
            output.append(
                {
                    "id": f"{revision}:{file_name}",
                    "name": file_name,
                    "size": item.get("size"),
                    "downloadUrl": f"https://huggingface.co/{quote(repo_id, safe='/')}/resolve/{quote(revision, safe='/')}/{quote(file_name, safe='/')}",
                    "metadata": item.get("metadata") or {},
                }
            )
        return output

    for branch in getattr(refs, "branches", []) or []:
        name = str(getattr(branch, "name", "") or "").strip()
        if not name or name == "main":
            continue
        versions.append({"id": name, "name": name, "createdAt": None, "files": version_files(name)})
    for tag in getattr(refs, "tags", []) or []:
        name = str(getattr(tag, "name", "") or "").strip()
        if not name:
            continue
        versions.append({"id": name, "name": name, "createdAt": None, "files": version_files(name)})
    return versions


def get_hf_model_detail(repo_id: str, token: Optional[str]) -> Dict[str, Any]:
    normalized_repo = str(repo_id or "").strip()
    cache_payload = {"source": "huggingface", "id": normalized_repo}

    def loader() -> Dict[str, Any]:
        api = HfApi(token=token)
        info = api.model_info(repo_id=normalized_repo, files_metadata=True)
        siblings = list(getattr(info, "siblings", None) or [])
        files: list[Dict[str, Any]] = []
        for sibling in siblings:
            file_name = str(getattr(sibling, "rfilename", "") or "")
            if not file_name:
                continue
            size_value = getattr(sibling, "size", None)
            if not isinstance(size_value, int):
                lfs = getattr(sibling, "lfs", None)
                if isinstance(lfs, dict):
                    size_value = lfs.get("size")
            files.append(
                {
                    "id": file_name,
                    "name": file_name,
                    "size": size_value if isinstance(size_value, int) and size_value > 0 else None,
                    "downloadUrl": f"https://huggingface.co/{quote(normalized_repo, safe='/')}/resolve/main/{quote(file_name, safe='/')}",
                    "metadata": {
                        "sha": getattr(sibling, "blob_id", None),
                        "lfs": getattr(sibling, "lfs", None),
                    },
                }
            )
        files = files[:200]
        readme_text = ""
        with contextlib.suppress(Exception):
            readme_path = hf_hub_download(
                repo_id=normalized_repo,
                filename="README.md",
                repo_type="model",
                token=token,
            )
            readme_text = Path(readme_path).read_text(encoding="utf-8", errors="replace")
        if not readme_text:
            card_data = getattr(info, "cardData", None)
            if isinstance(card_data, dict):
                readme_text = str(card_data.get("description") or "")
        preview_urls = collect_hf_preview_urls(normalized_repo, siblings)
        if not preview_urls:
            primary_preview = resolve_preview_url(info)
            if primary_preview:
                preview_urls = [primary_preview]
        tags = [str(tag) for tag in (getattr(info, "tags", None) or []) if str(tag or "").strip()]
        versions = hf_repo_ref_versions(api, normalized_repo, files)
        return {
            "source": "huggingface",
            "id": normalized_repo,
            "name": normalized_repo,
            "title": normalized_repo,
            "description": readme_text,
            "description_markdown": True,
            "tags": tags[:60],
            "previews": preview_urls[:12],
            "versions": versions[:20],
            "default_version_id": "main",
        }

    return cache_get_or_set("hf-detail", cache_payload, loader)


def get_civitai_model_detail(model_id: int) -> Dict[str, Any]:
    cache_payload = {"source": "civitai", "id": int(model_id)}

    def loader() -> Dict[str, Any]:
        payload = civitai_request_json(f"models/{int(model_id)}")
        name = str(payload.get("name") or f"civitai/{int(model_id)}")
        description = str(payload.get("description") or "")
        tags = [str(tag) for tag in (payload.get("tags") or []) if str(tag or "").strip()]
        versions_payload = payload.get("modelVersions", []) if isinstance(payload.get("modelVersions"), list) else []
        versions: list[Dict[str, Any]] = []
        previews: list[str] = []
        for version in versions_payload:
            if not isinstance(version, dict):
                continue
            version_id = safe_int(version.get("id"))
            version_name = str(version.get("name") or (version_id if version_id is not None else "version"))
            files_payload = version.get("files", []) if isinstance(version.get("files"), list) else []
            files: list[Dict[str, Any]] = []
            for file_entry in files_payload:
                if not isinstance(file_entry, dict):
                    continue
                file_id = safe_int(file_entry.get("id"))
                file_name = str(file_entry.get("name") or (file_id if file_id is not None else "file"))
                size_bytes = None
                size_kb = file_entry.get("sizeKB")
                if isinstance(size_kb, (int, float)) and size_kb > 0:
                    size_bytes = int(float(size_kb) * 1024)
                files.append(
                    {
                        "id": file_id if file_id is not None else file_name,
                        "name": file_name,
                        "size": size_bytes,
                        "downloadUrl": str(file_entry.get("downloadUrl") or ""),
                        "metadata": {
                            "type": file_entry.get("type"),
                            "format": file_entry.get("metadata", {}).get("format") if isinstance(file_entry.get("metadata"), dict) else None,
                            "fp": file_entry.get("metadata", {}).get("fp") if isinstance(file_entry.get("metadata"), dict) else None,
                        },
                    }
                )
            images_payload = version.get("images", []) if isinstance(version.get("images"), list) else []
            for image in images_payload:
                if not isinstance(image, dict):
                    continue
                image_url = str(image.get("url") or "").strip()
                if image_url:
                    previews.append(image_url)
            versions.append(
                {
                    "id": version_id if version_id is not None else version_name,
                    "name": version_name,
                    "createdAt": version.get("createdAt"),
                    "files": files[:80],
                }
            )
        unique_previews: list[str] = []
        seen: set[str] = set()
        for url in previews:
            if url in seen:
                continue
            seen.add(url)
            unique_previews.append(url)
        return {
            "source": "civitai",
            "id": f"civitai/{int(model_id)}",
            "name": name,
            "title": name,
            "description": description,
            "description_markdown": False,
            "tags": tags[:60],
            "previews": unique_previews[:12],
            "versions": versions[:30],
            "default_version_id": versions[0]["id"] if versions else None,
            "model_url": f"https://civitai.com/models/{int(model_id)}",
        }

    return cache_get_or_set("civitai-detail", cache_payload, loader)


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
    source: Optional[Literal["huggingface", "civitai"]] = None
    hf_revision: Optional[str] = None
    civitai_model_id: Optional[int] = Field(default=None, ge=1)
    civitai_version_id: Optional[int] = Field(default=None, ge=1)
    civitai_file_id: Optional[int] = Field(default=None, ge=1)


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


def select_civitai_download_file(
    model_payload: Dict[str, Any],
    version_id: Optional[int],
    file_id: Optional[int],
) -> tuple[Dict[str, Any], Dict[str, Any]]:
    versions = model_payload.get("modelVersions", [])
    if not isinstance(versions, list) or not versions:
        raise RuntimeError("No modelVersions found on CivitAI model")
    selected_version: Optional[Dict[str, Any]] = None
    if version_id:
        for version in versions:
            if not isinstance(version, dict):
                continue
            if safe_int(version.get("id")) == int(version_id):
                selected_version = version
                break
        if selected_version is None:
            raise RuntimeError(f"CivitAI model version not found: {version_id}")
    else:
        for version in versions:
            if isinstance(version, dict):
                selected_version = version
                break
    if selected_version is None:
        raise RuntimeError("No valid model version found")

    files = selected_version.get("files", [])
    if not isinstance(files, list) or not files:
        raise RuntimeError("No downloadable files found in selected CivitAI version")
    selected_file: Optional[Dict[str, Any]] = None
    if file_id:
        for file_entry in files:
            if not isinstance(file_entry, dict):
                continue
            if safe_int(file_entry.get("id")) == int(file_id):
                selected_file = file_entry
                break
        if selected_file is None:
            raise RuntimeError(f"CivitAI file not found in selected version: {file_id}")
    else:
        model_candidates: list[Dict[str, Any]] = []
        fallback_candidates: list[Dict[str, Any]] = []
        for file_entry in files:
            if not isinstance(file_entry, dict):
                continue
            if not str(file_entry.get("downloadUrl") or "").strip():
                continue
            kind = str(file_entry.get("type") or "").strip().lower()
            if kind in ("model", "checkpoint"):
                model_candidates.append(file_entry)
            else:
                fallback_candidates.append(file_entry)
        if model_candidates:
            selected_file = model_candidates[0]
        elif fallback_candidates:
            selected_file = fallback_candidates[0]
    if selected_file is None:
        raise RuntimeError("No downloadable CivitAI file with valid URL was found")
    return selected_version, selected_file


def download_preview_image(preview_url: str, destination: Path) -> None:
    req = UrlRequest(preview_url, headers={"User-Agent": "ROCm-VideoGen/1.0"})
    with contextlib.closing(urlopen(req, timeout=20)) as resp, destination.open("wb") as out_file:
        while True:
            chunk = resp.read(1024 * 256)
            if not chunk:
                break
            out_file.write(chunk)


def save_hf_model_metadata(
    model_dir: Path,
    repo_id: str,
    revision: str,
    token: Optional[str],
) -> None:
    api = HfApi(token=token)
    info = api.model_info(repo_id=repo_id, revision=revision, files_metadata=True)
    metadata = {
        "source": "huggingface",
        "repo_id": repo_id,
        "revision": revision,
        "model_url": f"https://huggingface.co/{quote(repo_id, safe='/')}",
        "downloads": getattr(info, "downloads", None),
        "likes": getattr(info, "likes", None),
    }
    (model_dir / "model_meta.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    preview_url = resolve_preview_url(info)
    if preview_url:
        preview_path = model_dir / "thumbnail.jpg"
        with contextlib.suppress(Exception):
            download_preview_image(preview_url, preview_path)


def download_model_worker(task_id: str, req: DownloadRequest) -> None:
    settings = settings_store.get()
    ensure_runtime_dirs(settings)
    repo_id = req.repo_id
    revision = (req.hf_revision or req.revision or "").strip() or "main"
    clean_target_dir = (req.target_dir or "").strip()
    base_dir_raw = clean_target_dir or settings["paths"]["models_dir"]
    models_dir = resolve_path(base_dir_raw)
    models_dir.mkdir(parents=True, exist_ok=True)
    token = settings["huggingface"].get("token") or None
    model_dir = models_dir / sanitize_repo_id(repo_id)
    LOGGER.info("download start task_id=%s repo=%s revision=%s target=%s", task_id, repo_id, revision or "main", str(model_dir))
    try:
        update_task(task_id, status="running", progress=0.0, message=f"Preparing download {repo_id}", downloaded_bytes=0, total_bytes=None)

        explicit_source = str(req.source or "").strip().lower()
        civitai_model_id = req.civitai_model_id or parse_civitai_model_id(repo_id)
        is_civitai = explicit_source == "civitai" or civitai_model_id is not None
        if civitai_model_id is not None:
            payload = civitai_request_json(f"models/{int(civitai_model_id)}")
            selected_version, selected_file = select_civitai_download_file(
                payload,
                version_id=req.civitai_version_id,
                file_id=req.civitai_file_id,
            )
            download_url = str(selected_file.get("downloadUrl") or "").strip()
            if not download_url:
                raise RuntimeError(f"Missing download URL on CivitAI model {civitai_model_id}")
            model_name = str(payload.get("name") or f"civitai-{civitai_model_id}")
            file_name = sanitize_download_filename(str(selected_file.get("name") or ""), f"civitai-{civitai_model_id}.safetensors")
            model_dir.mkdir(parents=True, exist_ok=True)
            file_path = model_dir / file_name
            total_bytes_hint: Optional[int] = None
            size_kb_raw = selected_file.get("sizeKB")
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
                "version_id": selected_version.get("id"),
                "version_name": selected_version.get("name"),
                "file_id": selected_file.get("id"),
                "file_name": file_name,
                "bytes": downloaded_bytes,
            }
            (model_dir / "civitai_model.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
            (model_dir / "model_meta.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
            images = selected_version.get("images", []) if isinstance(selected_version.get("images"), list) else []
            preview_url = ""
            for image in images:
                if isinstance(image, dict):
                    preview_url = str(image.get("url") or "").strip()
                    if preview_url:
                        break
            if preview_url:
                with contextlib.suppress(Exception):
                    download_preview_image(preview_url, model_dir / "thumbnail.jpg")
        else:
            if is_civitai:
                raise RuntimeError("CivitAI download requires civitai_model_id or repo_id like civitai/<id>")
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
            with contextlib.suppress(Exception):
                save_hf_model_metadata(model_dir=model_dir, repo_id=repo_id, revision=revision, token=token)
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


@app.get("/api/models/search2")
def search_models_v2(
    task: Literal["text-to-image", "image-to-image", "text-to-video", "image-to-video"],
    query: str = "",
    limit: int = 30,
    source: Literal["all", "huggingface", "civitai"] = "all",
    base_model: str = "",
    sort: Literal["popularity", "downloads", "likes", "updated", "created"] = "downloads",
    nsfw: Literal["include", "exclude"] = "exclude",
    page: int = 1,
    cursor: str = "",
    model_kind: str = "",
) -> Dict[str, Any]:
    settings = settings_store.get()
    token = settings["huggingface"].get("token") or None
    capped_limit = min(max(limit, 1), 100)
    current_page = max(int(page), 1)
    include_hf = source in ("all", "huggingface")
    include_civitai = source in ("all", "civitai") and task in ("text-to-image", "image-to-image")
    if cursor.strip():
        with contextlib.suppress(Exception):
            current_page = max(int(cursor.strip()), 1)

    hf_limit = 0
    civitai_limit = 0
    if include_hf and include_civitai:
        civitai_limit = max(1, capped_limit // 2)
        hf_limit = max(1, capped_limit - civitai_limit)
    elif include_hf:
        hf_limit = capped_limit
    elif include_civitai:
        civitai_limit = capped_limit

    hf_data = {"items": [], "has_next": False}
    civitai_data = {"items": [], "has_next": False}
    if hf_limit > 0:
        hf_data = search_hf_models_v2(
            task=task,
            query=query,
            limit=hf_limit,
            token=token,
            page=current_page,
            sort=sort,
            model_kind=model_kind,
        )
    if civitai_limit > 0:
        civitai_data = search_civitai_models_v2(
            task=task,
            query=query,
            limit=civitai_limit,
            page=current_page,
            sort=sort,
            nsfw=nsfw,
            model_kind=model_kind,
        )

    hf_items = list(hf_data.get("items") or [])
    civitai_items = list(civitai_data.get("items") or [])
    results: list[Dict[str, Any]] = []
    max_len = max(len(hf_items), len(civitai_items))
    for index in range(max_len):
        if index < len(hf_items):
            results.append(hf_items[index])
        if index < len(civitai_items):
            results.append(civitai_items[index])
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

    models_dir = resolve_path(settings["paths"]["models_dir"])
    installed = collect_installed_model_id_set(models_dir)
    for item in results:
        if not item.get("base_model"):
            item["base_model"] = infer_base_model_label(item.get("id"), item.get("pipeline_tag"), item.get("type"))
        model_id = normalize_model_lookup_key(str(item.get("id") or ""))
        item["installed"] = model_id in installed

    has_next = bool(hf_data.get("has_next")) or bool(civitai_data.get("has_next"))
    next_cursor = str(current_page + 1) if has_next else None
    prev_cursor = str(current_page - 1) if current_page > 1 else None
    return {
        "items": results,
        "next_cursor": next_cursor,
        "prev_cursor": prev_cursor,
        "page_info": {
            "page": current_page,
            "limit": capped_limit,
            "has_next": has_next,
            "has_prev": current_page > 1,
            "source": source,
            "task": task,
            "sort": sort,
            "nsfw": nsfw,
            "model_kind": normalized_model_kind(model_kind) or "checkpoint",
        },
    }


@app.get("/api/models/detail")
def model_detail(source: Literal["huggingface", "civitai"], id: str) -> Dict[str, Any]:
    settings = settings_store.get()
    token = settings["huggingface"].get("token") or None
    if source == "huggingface":
        repo_id = str(id or "").strip()
        if not repo_id:
            raise HTTPException(status_code=400, detail="id is required")
        try:
            return get_hf_model_detail(repo_id=repo_id, token=token)
        except Exception as exc:
            LOGGER.warning("hf detail failed id=%s", repo_id, exc_info=True)
            raise HTTPException(status_code=502, detail=f"Failed to fetch Hugging Face detail: {exc}") from exc
    model_id = parse_civitai_model_id(id) if str(id).lower().startswith("civitai/") else safe_int(id)
    if not model_id:
        raise HTTPException(status_code=400, detail="Invalid CivitAI id")
    try:
        return get_civitai_model_detail(model_id=int(model_id))
    except Exception as exc:
        LOGGER.warning("civitai detail failed id=%s", model_id, exc_info=True)
        raise HTTPException(status_code=502, detail=f"Failed to fetch CivitAI detail: {exc}") from exc


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
    LOGGER.info(
        "download requested task_id=%s repo=%s source=%s revision=%s hf_revision=%s civitai_model_id=%s civitai_version_id=%s civitai_file_id=%s target_dir=%s",
        task_id,
        req.repo_id,
        req.source or "",
        req.revision or "",
        req.hf_revision or "",
        req.civitai_model_id,
        req.civitai_version_id,
        req.civitai_file_id,
        req.target_dir,
    )
    thread = threading.Thread(
        target=download_model_worker,
        args=(task_id, req),
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

===== FILE: static/index.html =====

<!doctype html>
<html lang="ja">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>ROCm VideoGen</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Chakra+Petch:wght@400;500;700&family=IBM+Plex+Mono:wght@400;500&display=swap" rel="stylesheet">
  <link rel="stylesheet" href="/static/styles.css" />
</head>
<body>
  <div class="backdrop"></div>
  <main class="shell">
    <header class="top">
      <div>
        <h1 data-i18n="appTitle">ROCm VideoGen Studio</h1>
        <p data-i18n="appSubtitle">Text-to-Image / Image-to-Image / Text-to-Video / Image-to-Video</p>
      </div>
      <div class="top-right">
        <label class="lang-picker">
          <span data-i18n="languageLabel" class="has-help" data-help-key="helpLanguage" tabindex="0">Language</span>
          <select id="languageSelect">
            <option value="en">English</option>
            <option value="ja">日本語</option>
            <option value="es">Español</option>
            <option value="fr">Français</option>
            <option value="de">Deutsch</option>
            <option value="it">Italiano</option>
            <option value="pt">Português</option>
            <option value="ru">Русский</option>
            <option value="ar">العربية</option>
          </select>
        </label>
        <div id="runtimeInfo" class="runtime" data-i18n="runtimeLoading">runtime: loading...</div>
      </div>
    </header>

    <nav class="tabs">
      <button class="tab active" data-tab="text-image" data-i18n="tabTextToImage">Text to Image</button>
      <button class="tab" data-tab="image-image" data-i18n="tabImageToImage">Image to Image</button>
      <button class="tab" data-tab="text" data-i18n="tabTextToVideo">Text to Video</button>
      <button class="tab" data-tab="image" data-i18n="tabImageToVideo">Image to Video</button>
      <button class="tab" data-tab="models" data-i18n="tabModels">Model Search</button>
      <button class="tab" data-tab="local-models" data-i18n="tabLocalModels">Local Models</button>
      <button class="tab" data-tab="outputs" data-i18n="tabOutputs">Outputs</button>
      <button class="tab" data-tab="settings" data-i18n="tabSettings">Settings</button>
    </nav>

    <section class="panel active" id="panel-text-image">
      <form id="text2imageForm" class="form">
        <label>
          <span data-i18n="labelPrompt" class="has-help" data-help-key="helpPromptText2Image" tabindex="0">Prompt</span>
          <textarea id="t2iPrompt" required data-i18n-placeholder="placeholderT2IPrompt" placeholder="A portrait photo of a fox wearing a suit, studio light"></textarea>
        </label>
        <label>
          <span data-i18n="labelNegativePrompt" class="has-help" data-help-key="helpNegativePromptText2Image" tabindex="0">Negative Prompt</span>
          <textarea id="t2iNegative" data-i18n-placeholder="placeholderNegativePrompt" placeholder="low quality, blurry"></textarea>
        </label>
        <label>
          <span data-i18n="labelModelSelect" class="has-help" data-help-key="helpModelSelectText2Image" tabindex="0">Model Selection</span>
          <div class="input-actions model-select-actions">
            <select id="t2iModelSelect"></select>
            <div id="t2iModelPreview" class="model-picked-preview model-preview-inline"></div>
            <button id="refreshT2IModels" type="button" data-i18n="btnRefreshModels">Refresh</button>
          </div>
        </label>
        <label>
          <span data-i18n="labelLoraSelect" class="has-help" data-help-key="helpLoraSelectText2Image" tabindex="0">LoRA Selection</span>
          <div class="input-actions">
            <select id="t2iLoraSelect" multiple size="4"></select>
            <button id="refreshT2ILoras" type="button" data-i18n="btnRefreshLoras">Refresh LoRAs</button>
          </div>
        </label>
        <label>
          <span data-i18n="labelVaeSelect" class="has-help" data-help-key="helpVaeSelectText2Image" tabindex="0">VAE Selection</span>
          <div class="input-actions">
            <select id="t2iVaeSelect"></select>
            <button id="refreshT2IVaes" type="button" data-i18n="btnRefreshVaes">Refresh VAEs</button>
          </div>
        </label>
        <div class="grid">
          <label><span data-i18n="labelSteps" class="has-help" data-help-key="helpStepsText2Image" tabindex="0">Steps</span> <input id="t2iSteps" type="number" min="1" max="120" value="30" /></label>
          <label><span data-i18n="labelGuidance" class="has-help" data-help-key="helpGuidanceText2Image" tabindex="0">Guidance</span> <input id="t2iGuidance" type="number" min="0" max="30" step="0.1" value="7.5" /></label>
          <label><span data-i18n="labelLoraScale" class="has-help" data-help-key="helpLoraScale" tabindex="0">LoRA Scale</span> <input id="t2iLoraScale" type="number" min="0" max="2" step="0.05" value="1.0" /></label>
          <label><span data-i18n="labelWidth" class="has-help" data-help-key="helpWidthText2Image" tabindex="0">Width</span> <input id="t2iWidth" type="number" min="128" max="2048" step="64" value="512" /></label>
          <label><span data-i18n="labelHeight" class="has-help" data-help-key="helpHeightText2Image" tabindex="0">Height</span> <input id="t2iHeight" type="number" min="128" max="2048" step="64" value="512" /></label>
          <label><span data-i18n="labelSeed" class="has-help" data-help-key="helpSeedText2Image" tabindex="0">Seed</span> <input id="t2iSeed" type="number" min="0" data-i18n-placeholder="placeholderSeed" placeholder="random if empty" /></label>
        </div>
        <button type="submit" class="primary" data-i18n="btnGenerateTextImage">Generate Text Image</button>
      </form>
    </section>

    <section class="panel" id="panel-image-image">
      <form id="image2imageForm" class="form">
        <label>
          <span data-i18n="labelInputImage" class="has-help" data-help-key="helpInputImage" tabindex="0">Input Image</span>
          <input id="i2iImage" type="file" accept="image/*" required />
        </label>
        <label>
          <span data-i18n="labelPrompt" class="has-help" data-help-key="helpPromptImage2Image" tabindex="0">Prompt</span>
          <textarea id="i2iPrompt" required data-i18n-placeholder="placeholderI2IPrompt" placeholder="Refine this image into a cinematic poster style"></textarea>
        </label>
        <label>
          <span data-i18n="labelNegativePrompt" class="has-help" data-help-key="helpNegativePromptImage2Image" tabindex="0">Negative Prompt</span>
          <textarea id="i2iNegative" data-i18n-placeholder="placeholderI2INegativePrompt" placeholder="low quality, blurry"></textarea>
        </label>
        <label>
          <span data-i18n="labelModelSelect" class="has-help" data-help-key="helpModelSelectImage2Image" tabindex="0">Model Selection</span>
          <div class="input-actions model-select-actions">
            <select id="i2iModelSelect"></select>
            <div id="i2iModelPreview" class="model-picked-preview model-preview-inline"></div>
            <button id="refreshI2IModels" type="button" data-i18n="btnRefreshModels">Refresh</button>
          </div>
        </label>
        <label>
          <span data-i18n="labelLoraSelect" class="has-help" data-help-key="helpLoraSelectImage2Image" tabindex="0">LoRA Selection</span>
          <div class="input-actions">
            <select id="i2iLoraSelect" multiple size="4"></select>
            <button id="refreshI2ILoras" type="button" data-i18n="btnRefreshLoras">Refresh LoRAs</button>
          </div>
        </label>
        <label>
          <span data-i18n="labelVaeSelect" class="has-help" data-help-key="helpVaeSelectImage2Image" tabindex="0">VAE Selection</span>
          <div class="input-actions">
            <select id="i2iVaeSelect"></select>
            <button id="refreshI2IVaes" type="button" data-i18n="btnRefreshVaes">Refresh VAEs</button>
          </div>
        </label>
        <div class="grid">
          <label><span data-i18n="labelSteps" class="has-help" data-help-key="helpStepsImage2Image" tabindex="0">Steps</span> <input id="i2iSteps" type="number" min="1" max="120" value="30" /></label>
          <label><span data-i18n="labelGuidance" class="has-help" data-help-key="helpGuidanceImage2Image" tabindex="0">Guidance</span> <input id="i2iGuidance" type="number" min="0" max="30" step="0.1" value="7.5" /></label>
          <label><span data-i18n="labelStrength" class="has-help" data-help-key="helpStrengthImage2Image" tabindex="0">Strength</span> <input id="i2iStrength" type="number" min="0" max="1" step="0.05" value="0.8" /></label>
          <label><span data-i18n="labelLoraScale" class="has-help" data-help-key="helpLoraScale" tabindex="0">LoRA Scale</span> <input id="i2iLoraScale" type="number" min="0" max="2" step="0.05" value="1.0" /></label>
          <label><span data-i18n="labelWidth" class="has-help" data-help-key="helpWidthImage2Image" tabindex="0">Width</span> <input id="i2iWidth" type="number" min="128" max="2048" step="64" value="512" /></label>
          <label><span data-i18n="labelHeight" class="has-help" data-help-key="helpHeightImage2Image" tabindex="0">Height</span> <input id="i2iHeight" type="number" min="128" max="2048" step="64" value="512" /></label>
          <label><span data-i18n="labelSeed" class="has-help" data-help-key="helpSeedImage2Image" tabindex="0">Seed</span> <input id="i2iSeed" type="number" min="0" data-i18n-placeholder="placeholderSeed" placeholder="random if empty" /></label>
        </div>
        <button type="submit" class="primary" data-i18n="btnGenerateImageImage">Generate Image Image</button>
      </form>
    </section>

    <section class="panel" id="panel-text">
      <form id="text2videoForm" class="form">
        <label>
          <span data-i18n="labelPrompt" class="has-help" data-help-key="helpPromptText2Video" tabindex="0">Prompt</span>
          <textarea id="t2vPrompt" required data-i18n-placeholder="placeholderT2VPrompt" placeholder="A cinematic drone shot above neon city..."></textarea>
        </label>
        <label>
          <span data-i18n="labelNegativePrompt" class="has-help" data-help-key="helpNegativePromptText2Video" tabindex="0">Negative Prompt</span>
          <textarea id="t2vNegative" data-i18n-placeholder="placeholderNegativePrompt" placeholder="low quality, blurry"></textarea>
        </label>
        <label>
          <span data-i18n="labelModelSelect" class="has-help" data-help-key="helpModelSelectText2Video" tabindex="0">Model Selection</span>
          <div class="input-actions model-select-actions">
            <select id="t2vModelSelect"></select>
            <div id="t2vModelPreview" class="model-picked-preview model-preview-inline"></div>
            <button id="refreshT2VModels" type="button" data-i18n="btnRefreshModels">Refresh</button>
          </div>
        </label>
        <label>
          <span data-i18n="labelLoraSelect" class="has-help" data-help-key="helpLoraSelectText2Video" tabindex="0">LoRA Selection</span>
          <div class="input-actions">
            <select id="t2vLoraSelect" multiple size="4"></select>
            <button id="refreshT2VLoras" type="button" data-i18n="btnRefreshLoras">Refresh LoRAs</button>
          </div>
        </label>
        <div class="grid">
          <label><span data-i18n="labelSteps" class="has-help" data-help-key="helpStepsText2Video" tabindex="0">Steps</span> <input id="t2vSteps" type="number" min="1" max="120" value="30" /></label>
          <label><span data-i18n="labelFrames" class="has-help" data-help-key="helpFramesText2Video" tabindex="0">Frames</span> <input id="t2vFrames" type="number" min="8" max="128" value="16" /></label>
          <label><span data-i18n="labelGuidance" class="has-help" data-help-key="helpGuidanceText2Video" tabindex="0">Guidance</span> <input id="t2vGuidance" type="number" min="0" max="30" step="0.1" value="9.0" /></label>
          <label><span data-i18n="labelLoraScale" class="has-help" data-help-key="helpLoraScale" tabindex="0">LoRA Scale</span> <input id="t2vLoraScale" type="number" min="0" max="2" step="0.05" value="1.0" /></label>
          <label><span data-i18n="labelFps" class="has-help" data-help-key="helpFpsText2Video" tabindex="0">FPS</span> <input id="t2vFps" type="number" min="1" max="60" value="8" /></label>
          <label>
            <span data-i18n="labelT2VBackend" class="has-help" data-help-key="helpT2VBackend" tabindex="0">T2V Backend</span>
            <select id="t2vBackendSelect">
              <option value="auto" data-i18n="backendAuto">Auto</option>
              <option value="cuda" data-i18n="backendCuda">GPU (CUDA/ROCm)</option>
              <option value="npu" data-i18n="backendNpu">NPU</option>
            </select>
          </label>
          <label><span data-i18n="labelSeed" class="has-help" data-help-key="helpSeedText2Video" tabindex="0">Seed</span> <input id="t2vSeed" type="number" min="0" data-i18n-placeholder="placeholderSeed" placeholder="random if empty" /></label>
        </div>
        <button type="submit" class="primary" data-i18n="btnGenerateTextVideo">Generate Text Video</button>
      </form>
    </section>

    <section class="panel" id="panel-image">
      <form id="image2videoForm" class="form">
        <label>
          <span data-i18n="labelInputImage" class="has-help" data-help-key="helpInputImage" tabindex="0">Input Image</span>
          <input id="i2vImage" type="file" accept="image/*" required />
        </label>
        <label>
          <span data-i18n="labelPrompt" class="has-help" data-help-key="helpPromptImage2Video" tabindex="0">Prompt</span>
          <textarea id="i2vPrompt" required data-i18n-placeholder="placeholderI2VPrompt" placeholder="Turn this image into a smooth cinematic motion..."></textarea>
        </label>
        <label>
          <span data-i18n="labelNegativePrompt" class="has-help" data-help-key="helpNegativePromptImage2Video" tabindex="0">Negative Prompt</span>
          <textarea id="i2vNegative" data-i18n-placeholder="placeholderI2VNegativePrompt" placeholder="artifact, flicker"></textarea>
        </label>
        <label>
          <span data-i18n="labelModelSelect" class="has-help" data-help-key="helpModelSelectImage2Video" tabindex="0">Model Selection</span>
          <div class="input-actions model-select-actions">
            <select id="i2vModelSelect"></select>
            <div id="i2vModelPreview" class="model-picked-preview model-preview-inline"></div>
            <button id="refreshI2VModels" type="button" data-i18n="btnRefreshModels">Refresh</button>
          </div>
        </label>
        <label>
          <span data-i18n="labelLoraSelect" class="has-help" data-help-key="helpLoraSelectImage2Video" tabindex="0">LoRA Selection</span>
          <div class="input-actions">
            <select id="i2vLoraSelect" multiple size="4"></select>
            <button id="refreshI2VLoras" type="button" data-i18n="btnRefreshLoras">Refresh LoRAs</button>
          </div>
        </label>
        <div class="grid">
          <label><span data-i18n="labelSteps" class="has-help" data-help-key="helpStepsImage2Video" tabindex="0">Steps</span> <input id="i2vSteps" type="number" min="1" max="120" value="30" /></label>
          <label><span data-i18n="labelFrames" class="has-help" data-help-key="helpFramesImage2Video" tabindex="0">Frames</span> <input id="i2vFrames" type="number" min="8" max="128" value="16" /></label>
          <label><span data-i18n="labelGuidance" class="has-help" data-help-key="helpGuidanceImage2Video" tabindex="0">Guidance</span> <input id="i2vGuidance" type="number" min="0" max="30" step="0.1" value="9.0" /></label>
          <label><span data-i18n="labelLoraScale" class="has-help" data-help-key="helpLoraScale" tabindex="0">LoRA Scale</span> <input id="i2vLoraScale" type="number" min="0" max="2" step="0.05" value="1.0" /></label>
          <label><span data-i18n="labelFps" class="has-help" data-help-key="helpFpsImage2Video" tabindex="0">FPS</span> <input id="i2vFps" type="number" min="1" max="60" value="8" /></label>
          <label><span data-i18n="labelWidth" class="has-help" data-help-key="helpWidthImage2Video" tabindex="0">Width</span> <input id="i2vWidth" type="number" min="128" max="2048" step="64" value="512" /></label>
          <label><span data-i18n="labelHeight" class="has-help" data-help-key="helpHeightImage2Video" tabindex="0">Height</span> <input id="i2vHeight" type="number" min="128" max="2048" step="64" value="512" /></label>
          <label><span data-i18n="labelSeed" class="has-help" data-help-key="helpSeedImage2Video" tabindex="0">Seed</span> <input id="i2vSeed" type="number" min="0" data-i18n-placeholder="placeholderSeed" placeholder="random if empty" /></label>
        </div>
        <button type="submit" class="primary" data-i18n="btnGenerateImageVideo">Generate Image Video</button>
      </form>
    </section>

    <section class="panel" id="panel-models">
      <form id="searchForm" class="form compact model-search-topbar">
        <label>
          <span data-i18n="labelDownloadSavePathOptional" class="has-help" data-help-key="helpDownloadSavePath" tabindex="0">Download Save Path (optional)</span>
          <input id="downloadTargetDir" data-i18n-placeholder="placeholderDownloadSavePath" placeholder="empty = use Models Directory from Settings" />
        </label>
        <div class="model-search-query-row">
          <input id="searchQuery" data-i18n-placeholder="placeholderSearchQuery" placeholder="i2vgen, text-to-video..." />
          <button type="submit" data-i18n="btnSearchModels">Search Models</button>
        </div>
        <div class="model-search-control-grid">
          <label>
            <span data-i18n="labelTask" class="has-help" data-help-key="helpTask" tabindex="0">Task</span>
            <select id="searchTask">
              <option value="text-to-image">text-to-image</option>
              <option value="image-to-image">image-to-image</option>
              <option value="text-to-video">text-to-video</option>
              <option value="image-to-video">image-to-video</option>
            </select>
          </label>
          <label>
            <span data-i18n="labelSearchSource" class="has-help" data-help-key="helpSearchSource" tabindex="0">Source</span>
            <select id="searchSource">
              <option value="all" data-i18n="searchSourceAll">All</option>
              <option value="huggingface" data-i18n="searchSourceHf">Hugging Face</option>
              <option value="civitai" data-i18n="searchSourceCivitai">CivitAI</option>
            </select>
          </label>
          <label>
            <span data-i18n="labelSearchBaseModel" class="has-help" data-help-key="helpSearchBaseModel" tabindex="0">Base Model</span>
            <select id="searchBaseModel"></select>
          </label>
          <label>
            <span data-i18n="labelSearchSort" class="has-help" data-help-key="helpSearchSort" tabindex="0">Sort</span>
            <select id="searchSort">
              <option value="downloads">downloads</option>
              <option value="likes">likes</option>
              <option value="updated">updated</option>
              <option value="created">created</option>
              <option value="popularity">popularity</option>
            </select>
          </label>
          <label>
            <span data-i18n="labelSearchNsfw" class="has-help" data-help-key="helpSearchNsfw" tabindex="0">NSFW</span>
            <select id="searchNsfw">
              <option value="exclude">exclude</option>
              <option value="include">include</option>
            </select>
          </label>
          <label>
            <span data-i18n="labelLimit" class="has-help" data-help-key="helpLimit" tabindex="0">Limit</span>
            <input id="searchLimit" type="number" min="1" max="100" value="30" />
          </label>
        </div>
      </form>
      <div id="searchResults" class="model-browser">
        <aside id="searchFilters" class="model-filter-pane">
          <h4 data-i18n="labelSearchModelKind">Model Kind</h4>
          <label>
            <span data-i18n="labelSearchModelKind" class="has-help" data-help-key="helpSearchModelKind" tabindex="0">Model Kind</span>
            <select id="searchModelKind">
              <option value="">auto</option>
              <option value="checkpoint">checkpoint</option>
              <option value="lora">lora</option>
              <option value="vae">vae</option>
              <option value="controlnet">controlnet</option>
              <option value="embedding">embedding</option>
              <option value="upscaler">upscaler</option>
            </select>
          </label>
          <label>
            <span data-i18n="labelSearchViewMode" class="has-help" data-help-key="helpSearchViewMode" tabindex="0">View</span>
            <select id="searchViewMode">
              <option value="grid">grid</option>
              <option value="list">list</option>
            </select>
          </label>
          <div class="model-filter-note" id="searchFilterNote"></div>
        </aside>
        <section class="model-results-pane">
          <div id="searchCards" class="model-card-grid"></div>
          <div class="model-pagination">
            <button id="searchPrevBtn" type="button">Prev</button>
            <span id="searchPageInfo">page 1</span>
            <button id="searchNextBtn" type="button">Next</button>
          </div>
        </section>
        <aside id="modelDetailPane" class="model-detail-pane">
          <div id="modelDetailEmpty" class="model-detail-empty" data-i18n="msgModelDetailEmpty">Select a model to view detail.</div>
          <div id="modelDetailContent" class="model-detail-content"></div>
        </aside>
      </div>
    </section>

    <section class="panel" id="panel-local-models">
      <h3 data-i18n="headingLocalModels">Local Models</h3>
      <label>
        <span data-i18n="labelLocalModelsPath" class="has-help" data-help-key="helpLocalModelsPath" tabindex="0">Local Models Path</span>
        <input id="localModelsDir" data-i18n-placeholder="placeholderLocalModelsPath" placeholder="empty = use Models Directory from Settings" />
      </label>
      <label>
        <span data-i18n="labelLocalLineage" class="has-help" data-help-key="helpLocalLineage" tabindex="0">Lineage</span>
        <select id="localLineageFilter"></select>
      </label>
      <button id="refreshLocalModels" type="button" data-i18n="btnRefreshLocalList">Refresh Local List</button>
      <div id="localModels" class="list"></div>
    </section>

    <section class="panel" id="panel-outputs">
      <h3 data-i18n="headingOutputs">Outputs</h3>
      <span id="outputsPath"></span>
      <button id="refreshOutputs" type="button" data-i18n="btnRefreshOutputs">Refresh Outputs</button>
      <div id="outputsList" class="list"></div>
    </section>

    <section class="panel" id="panel-settings">
      <form id="settingsForm" class="form">
        <div class="settings-stack">
          <label>
            <span data-i18n="labelModelsDirectory" class="has-help" data-help-key="helpModelsDirectory" tabindex="0">Models Directory</span>
            <input id="cfgModelsDir" />
          </label>
          <label><span data-i18n="labelListenPort" class="has-help" data-help-key="helpListenPort" tabindex="0">Listen Port</span> <input id="cfgListenPort" type="number" min="1" max="65535" step="1" /></label>
          <label>
            <span data-i18n="labelRocmAotriton" class="has-help" data-help-key="helpRocmAotriton" tabindex="0">ROCm AOTriton Experimental</span>
            <input id="cfgRocmAotriton" type="checkbox" />
          </label>
          <label><span data-i18n="labelOutputsDirectory" class="has-help" data-help-key="helpOutputsDirectory" tabindex="0">Outputs Directory</span> <input id="cfgOutputsDir" /></label>
          <label><span data-i18n="labelTempDirectory" class="has-help" data-help-key="helpTempDirectory" tabindex="0">Temp Directory</span> <input id="cfgTmpDir" /></label>
          <label>
            <span data-i18n="labelLogLevel" class="has-help" data-help-key="helpLogLevel" tabindex="0">Log Level</span>
            <select id="cfgLogLevel">
              <option value="DEBUG" data-i18n="logLevelDebug">DEBUG</option>
              <option value="INFO" data-i18n="logLevelInfo">INFO</option>
              <option value="WARNING" data-i18n="logLevelWarning">WARNING</option>
              <option value="ERROR" data-i18n="logLevelError">ERROR</option>
            </select>
          </label>
          <label><span data-i18n="labelHfToken" class="has-help" data-help-key="helpHfToken" tabindex="0">HF Token</span> <input id="cfgToken" type="password" data-i18n-placeholder="placeholderOptional" placeholder="optional" /></label>
          <label>
            <span data-i18n="labelDefaultT2VBackend" class="has-help" data-help-key="helpDefaultT2VBackend" tabindex="0">Default T2V Backend</span>
            <select id="cfgT2VBackend">
              <option value="auto" data-i18n="backendAuto">Auto</option>
              <option value="cuda" data-i18n="backendCuda">GPU (CUDA/ROCm)</option>
              <option value="npu" data-i18n="backendNpu">NPU</option>
            </select>
          </label>
          <label>
            <span data-i18n="labelT2VNpuRunner" class="has-help" data-help-key="helpT2VNpuRunner" tabindex="0">T2V NPU Runner</span>
            <input id="cfgT2VNpuRunner" data-i18n-placeholder="placeholderOptional" placeholder="optional" />
          </label>
          <label>
            <span data-i18n="labelT2VNpuModelDir" class="has-help" data-help-key="helpT2VNpuModelDir" tabindex="0">T2V NPU Model Directory</span>
            <input id="cfgT2VNpuModelDir" data-i18n-placeholder="placeholderOptional" placeholder="optional" />
          </label>
          <label><span data-i18n="labelDefaultTextModel" class="has-help" data-help-key="helpDefaultTextModel" tabindex="0">Default: Text to Video</span> <select id="cfgTextModel"></select></label>
          <label><span data-i18n="labelDefaultImageModel" class="has-help" data-help-key="helpDefaultImageModel" tabindex="0">Default: Image to Video</span> <select id="cfgImageModel"></select></label>
          <label><span data-i18n="labelDefaultTextImageModel" class="has-help" data-help-key="helpDefaultTextImageModel" tabindex="0">Default: Text to Image</span> <select id="cfgTextImageModel"></select></label>
          <label><span data-i18n="labelDefaultImageImageModel" class="has-help" data-help-key="helpDefaultImageImageModel" tabindex="0">Default: Image to Image</span> <select id="cfgImageImageModel"></select></label>
          <label><span data-i18n="labelDefaultSteps" class="has-help" data-help-key="helpDefaultSteps" tabindex="0">Default Steps</span> <input id="cfgSteps" type="number" min="1" max="120" /></label>
          <label><span data-i18n="labelDefaultFrames" class="has-help" data-help-key="helpDefaultFrames" tabindex="0">Default Frames</span> <input id="cfgFrames" type="number" min="8" max="128" /></label>
          <label><span data-i18n="labelDefaultGuidance" class="has-help" data-help-key="helpDefaultGuidance" tabindex="0">Default Guidance</span> <input id="cfgGuidance" type="number" min="0" max="30" step="0.1" /></label>
          <label><span data-i18n="labelDefaultFps" class="has-help" data-help-key="helpDefaultFps" tabindex="0">Default FPS</span> <input id="cfgFps" type="number" min="1" max="60" /></label>
          <label><span data-i18n="labelDefaultWidth" class="has-help" data-help-key="helpDefaultWidth" tabindex="0">Default Width</span> <input id="cfgWidth" type="number" min="128" max="2048" step="64" /></label>
          <label><span data-i18n="labelDefaultHeight" class="has-help" data-help-key="helpDefaultHeight" tabindex="0">Default Height</span> <input id="cfgHeight" type="number" min="128" max="2048" step="64" /></label>
          <label>
            <span data-i18n="labelClearHfCache" class="has-help" data-help-key="helpClearHfCache" tabindex="0">Clear Hugging Face Cache</span>
            <button id="clearHfCacheBtn" type="button" data-i18n="btnClearHfCache">Clear Cache</button>
          </label>
        </div>
        <button type="submit" class="primary" data-i18n="btnSaveSettings">Save Settings</button>
      </form>
    </section>

    <section class="status">
      <div id="taskStatus" data-i18n="statusNoTask">No task running.</div>
      <div id="taskProgressWrap" class="task-progress-wrap" aria-live="polite">
        <div id="taskProgressBar" class="task-progress-bar"></div>
        <div id="taskProgressLabel" class="task-progress-label">0% | ETA --:-- | ELAPSED 00:00</div>
      </div>
      <img id="imagePreview" alt="Generated image preview" />
      <video id="preview" controls></video>
    </section>
  </main>
  <script src="/static/app.js"></script>
</body>
</html>

===== FILE: static/styles.css =====

:root {
  --bg-1: #091018;
  --bg-2: #101e2c;
  --card: rgba(9, 16, 24, 0.72);
  --line: rgba(114, 181, 223, 0.4);
  --text: #f1f7ff;
  --sub: #9cb7cc;
  --accent: #14c5af;
  --accent-2: #ff9a3c;
}

* {
  box-sizing: border-box;
}

body {
  margin: 0;
  font-family: "Chakra Petch", sans-serif;
  color: var(--text);
  min-height: 100vh;
  background: radial-gradient(circle at 20% 10%, #18354b 0%, transparent 40%),
    radial-gradient(circle at 80% 70%, #3b2200 0%, transparent 35%),
    linear-gradient(140deg, var(--bg-1), var(--bg-2));
}

.backdrop {
  position: fixed;
  inset: 0;
  pointer-events: none;
  background-image: linear-gradient(rgba(20, 197, 175, 0.07) 1px, transparent 1px),
    linear-gradient(90deg, rgba(20, 197, 175, 0.07) 1px, transparent 1px);
  background-size: 32px 32px;
}

.shell {
  width: min(1180px, 94vw);
  margin: 2rem auto;
  padding: 1.2rem;
  border: 1px solid var(--line);
  border-radius: 18px;
  backdrop-filter: blur(7px);
  background: var(--card);
  animation: rise 450ms ease forwards;
}

@keyframes rise {
  from {
    opacity: 0;
    transform: translateY(12px);
  }
  to {
    opacity: 1;
    transform: translateY(0);
  }
}

.top {
  display: flex;
  gap: 1rem;
  justify-content: space-between;
  align-items: flex-start;
}

.top-right {
  display: flex;
  flex-direction: column;
  gap: 0.5rem;
  align-items: flex-end;
}

.lang-picker {
  display: flex;
  gap: 0.5rem;
  align-items: center;
  font-size: 0.9rem;
}

.lang-picker select {
  min-width: 160px;
}

h1 {
  margin: 0 0 0.3rem;
  font-size: clamp(1.4rem, 3.5vw, 2rem);
}

.top p {
  margin: 0;
  color: var(--sub);
}

.runtime {
  max-width: 52ch;
  font-family: "IBM Plex Mono", monospace;
  font-size: 0.85rem;
  color: #d2e4f2;
  border: 1px solid rgba(255, 154, 60, 0.45);
  border-radius: 10px;
  padding: 0.6rem;
}

.tabs {
  display: flex;
  flex-wrap: wrap;
  gap: 0.6rem;
  margin: 1rem 0;
}

.tab {
  border: 1px solid var(--line);
  border-radius: 999px;
  padding: 0.45rem 0.85rem;
  background: transparent;
  color: var(--text);
  cursor: pointer;
}

.tab.active {
  border-color: var(--accent);
  background: rgba(20, 197, 175, 0.2);
}

.panel {
  display: none;
  padding: 1rem;
  border: 1px solid var(--line);
  border-radius: 12px;
  margin-bottom: 1rem;
  background: rgba(5, 10, 18, 0.6);
}

.panel.active {
  display: block;
  animation: rise 250ms ease forwards;
}

.form {
  display: flex;
  flex-direction: column;
  gap: 0.8rem;
}

.compact {
  margin-bottom: 0.8rem;
}

label {
  display: flex;
  flex-direction: column;
  gap: 0.3rem;
  font-size: 0.95rem;
}

.has-help {
  position: relative;
  cursor: help;
  text-decoration: underline dotted rgba(156, 183, 204, 0.9);
  text-underline-offset: 2px;
}

.has-help:hover::after,
.has-help:focus-visible::after {
  content: attr(data-help);
  position: absolute;
  left: 0;
  top: calc(100% + 8px);
  width: min(480px, 78vw);
  white-space: pre-wrap;
  z-index: 40;
  padding: 0.55rem 0.65rem;
  border-radius: 10px;
  border: 1px solid rgba(130, 174, 208, 0.55);
  background: rgba(5, 10, 18, 0.96);
  color: #e7f2ff;
  font-family: "IBM Plex Mono", monospace;
  font-size: 0.78rem;
  line-height: 1.45;
  box-shadow: 0 8px 22px rgba(0, 0, 0, 0.35);
}

.input-actions {
  display: flex;
  gap: 0.5rem;
}

.input-actions input,
.input-actions select {
  flex: 1;
}

.input-actions select[multiple] {
  min-height: 6.8rem;
}

.model-select-actions {
  align-items: center;
}

.model-select-actions select {
  min-width: 260px;
}

input,
textarea,
select,
button {
  font: inherit;
}

input,
textarea,
select {
  border: 1px solid rgba(130, 174, 208, 0.4);
  border-radius: 10px;
  padding: 0.55rem 0.65rem;
  color: var(--text);
  background: rgba(4, 8, 15, 0.8);
}

textarea {
  min-height: 86px;
  resize: vertical;
}

.grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
  gap: 0.7rem;
}

.settings-stack {
  display: flex;
  flex-direction: column;
  gap: 0.7rem;
}

button {
  border: 1px solid rgba(255, 154, 60, 0.8);
  border-radius: 10px;
  padding: 0.55rem 0.9rem;
  color: var(--text);
  background: rgba(255, 154, 60, 0.1);
  cursor: pointer;
}

button:hover {
  filter: brightness(1.08);
}

.downloaded-flag {
  display: inline-flex;
  align-items: center;
  border: 1px solid rgba(20, 197, 175, 0.8);
  border-radius: 999px;
  padding: 0.35rem 0.75rem;
  color: #8ff3de;
  background: rgba(20, 197, 175, 0.12);
  font-size: 0.85rem;
  white-space: nowrap;
}

.local-delete-btn {
  border-color: rgba(255, 107, 107, 0.9);
  background: rgba(255, 107, 107, 0.12);
}

.output-delete-btn {
  border-color: rgba(255, 107, 107, 0.9);
  background: rgba(255, 107, 107, 0.12);
}

.primary {
  border-color: rgba(20, 197, 175, 0.9);
  background: rgba(20, 197, 175, 0.17);
}

.list {
  display: flex;
  flex-direction: column;
  gap: 0.55rem;
}

.model-search-topbar {
  margin-bottom: 0.7rem;
}

.model-search-query-row {
  display: grid;
  grid-template-columns: 1fr auto;
  gap: 0.55rem;
  align-items: end;
}

.model-search-control-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(170px, 1fr));
  gap: 0.55rem 0.75rem;
  align-items: end;
}

.model-browser {
  display: grid;
  grid-template-columns: minmax(190px, 0.9fr) minmax(420px, 2fr) minmax(280px, 1.15fr);
  gap: 0.7rem;
  align-items: start;
}

.model-filter-pane,
.model-results-pane,
.model-detail-pane {
  border: 1px solid rgba(130, 174, 208, 0.35);
  border-radius: 10px;
  background: rgba(4, 8, 15, 0.45);
  padding: 0.65rem;
}

.model-filter-pane h4 {
  margin: 0 0 0.55rem 0;
  color: #bfe8ff;
}

.model-filter-pane {
  display: flex;
  flex-direction: column;
  gap: 0.55rem;
  position: sticky;
  top: 0.5rem;
}

.model-filter-note {
  color: var(--sub);
  font-size: 0.8rem;
  border-top: 1px dashed rgba(130, 174, 208, 0.35);
  padding-top: 0.45rem;
}

.model-card-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(230px, 1fr));
  gap: 0.65rem;
}

.model-card-grid.list-mode {
  grid-template-columns: 1fr;
}

.model-card {
  border: 1px solid rgba(130, 174, 208, 0.35);
  border-radius: 10px;
  background: rgba(3, 7, 14, 0.75);
  overflow: hidden;
  display: flex;
  flex-direction: column;
  min-height: 320px;
}

.model-card.list-mode {
  display: grid;
  grid-template-columns: 180px 1fr;
  min-height: 160px;
}

.model-card-cover {
  width: 100%;
  height: 150px;
  object-fit: cover;
  border-bottom: 1px solid rgba(130, 174, 208, 0.35);
  background: rgba(4, 8, 15, 0.9);
}

.model-card.list-mode .model-card-cover {
  height: 100%;
  min-height: 160px;
  border-bottom: none;
  border-right: 1px solid rgba(130, 174, 208, 0.35);
}

.model-card-body {
  display: flex;
  flex-direction: column;
  gap: 0.45rem;
  padding: 0.6rem;
}

.model-card-title {
  color: #c7efff;
  font-size: 0.94rem;
  font-weight: 700;
  word-break: break-word;
  margin: 0;
}

.model-card-id {
  color: var(--sub);
  font-family: "IBM Plex Mono", monospace;
  font-size: 0.77rem;
  word-break: break-all;
}

.model-meta-line {
  color: var(--sub);
  font-size: 0.78rem;
  font-family: "IBM Plex Mono", monospace;
  word-break: break-word;
}

.model-card-actions {
  display: flex;
  flex-wrap: wrap;
  gap: 0.35rem;
}

.model-card-actions button,
.model-card-actions a {
  font-size: 0.79rem;
}

.model-status-badge {
  display: inline-flex;
  align-items: center;
  border-radius: 999px;
  border: 1px solid rgba(130, 174, 208, 0.45);
  padding: 0.22rem 0.52rem;
  font-size: 0.73rem;
  font-family: "IBM Plex Mono", monospace;
  color: #d7eaf9;
  background: rgba(130, 174, 208, 0.12);
}

.model-status-badge.downloaded {
  border-color: rgba(20, 197, 175, 0.8);
  color: #8ff3de;
  background: rgba(20, 197, 175, 0.12);
}

.model-pagination {
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 0.65rem;
  margin-top: 0.65rem;
}

.model-pagination span {
  color: var(--sub);
  font-size: 0.82rem;
  font-family: "IBM Plex Mono", monospace;
}

.model-detail-pane {
  min-height: 320px;
  max-height: 78vh;
  overflow: auto;
}

.model-detail-empty {
  color: var(--sub);
  font-size: 0.9rem;
}

.model-detail-content {
  display: none;
  flex-direction: column;
  gap: 0.6rem;
}

.model-detail-content.active {
  display: flex;
}

.model-detail-head {
  display: flex;
  flex-direction: column;
  gap: 0.25rem;
}

.model-detail-title {
  margin: 0;
  color: #d9f3ff;
  font-size: 1rem;
  word-break: break-word;
}

.model-detail-id {
  color: var(--sub);
  font-size: 0.8rem;
  font-family: "IBM Plex Mono", monospace;
  word-break: break-all;
}

.model-detail-meta {
  color: var(--sub);
  font-size: 0.8rem;
  font-family: "IBM Plex Mono", monospace;
}

.model-detail-text {
  border: 1px solid rgba(130, 174, 208, 0.35);
  border-radius: 8px;
  padding: 0.55rem;
  background: rgba(4, 8, 15, 0.6);
  max-height: 220px;
  overflow: auto;
  white-space: pre-wrap;
  color: #d7eaf9;
  font-size: 0.84rem;
}

.model-detail-tags {
  display: flex;
  flex-wrap: wrap;
  gap: 0.35rem;
}

.model-detail-tag {
  font-size: 0.73rem;
  border: 1px solid rgba(130, 174, 208, 0.45);
  border-radius: 999px;
  padding: 0.18rem 0.5rem;
  color: #c9e8ff;
}

.model-detail-gallery {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(98px, 1fr));
  gap: 0.45rem;
}

.model-detail-gallery img {
  width: 100%;
  height: 70px;
  object-fit: cover;
  border-radius: 7px;
  border: 1px solid rgba(130, 174, 208, 0.35);
}

.model-detail-grid {
  display: grid;
  grid-template-columns: 1fr;
  gap: 0.5rem;
}

.model-detail-actions {
  display: flex;
  gap: 0.4rem;
  flex-wrap: wrap;
}

.row {
  display: flex;
  justify-content: space-between;
  gap: 0.7rem;
  border: 1px solid rgba(130, 174, 208, 0.35);
  border-radius: 10px;
  padding: 0.65rem;
  align-items: center;
}

.row > div {
  display: flex;
  flex-direction: column;
  gap: 0.2rem;
  min-width: 0;
}

.model-row {
  align-items: flex-start;
}

.local-actions {
  display: flex;
  gap: 0.4rem;
  flex-wrap: wrap;
  justify-content: flex-end;
}

.model-main {
  display: flex;
  gap: 0.7rem;
  min-width: 0;
}

.model-preview {
  width: 96px;
  height: 54px;
  object-fit: cover;
  border-radius: 8px;
  border: 1px solid rgba(130, 174, 208, 0.35);
  background: rgba(4, 8, 15, 0.8);
}

.model-main a {
  color: #b6e7ff;
  text-decoration: none;
}

.model-main a:hover {
  text-decoration: underline;
}

.row span {
  color: var(--sub);
  font-size: 0.86rem;
  font-family: "IBM Plex Mono", monospace;
  word-break: break-all;
}

#outputsPath {
  display: block;
  color: var(--sub);
  font-size: 0.84rem;
  margin-bottom: 0.55rem;
  font-family: "IBM Plex Mono", monospace;
  word-break: break-all;
}

.status {
  border: 1px solid var(--line);
  border-radius: 12px;
  padding: 0.9rem;
  background: rgba(5, 10, 18, 0.6);
}

.path-browser {
  margin-top: -0.2rem;
}

.path-browser-head {
  display: flex;
  justify-content: space-between;
  align-items: center;
  gap: 0.7rem;
}

.path-browser-head h3 {
  margin: 0;
}

.path-browser-current {
  display: flex;
  flex-direction: column;
  gap: 0.3rem;
}

.path-browser-current code {
  display: block;
  padding: 0.5rem 0.6rem;
  border: 1px solid rgba(130, 174, 208, 0.35);
  border-radius: 8px;
  font-family: "IBM Plex Mono", monospace;
  color: #c8e5f7;
  word-break: break-all;
}

.path-browser-actions {
  display: flex;
  gap: 0.6rem;
  flex-wrap: wrap;
}

.path-browser-roots {
  display: flex;
  flex-wrap: wrap;
  gap: 0.5rem;
}

.path-btn {
  border: 1px solid rgba(130, 174, 208, 0.45);
  border-radius: 8px;
  padding: 0.4rem 0.55rem;
  background: rgba(4, 8, 15, 0.6);
  color: var(--text);
  cursor: pointer;
  font-family: "IBM Plex Mono", monospace;
  font-size: 0.8rem;
}

.path-browser-list-row {
  display: flex;
  justify-content: space-between;
  gap: 0.6rem;
  align-items: center;
}

.path-browser-list-row span {
  color: var(--sub);
  font-family: "IBM Plex Mono", monospace;
  font-size: 0.82rem;
  word-break: break-all;
}

.model-picked-preview {
  border: 1px solid rgba(130, 174, 208, 0.35);
  border-radius: 10px;
  padding: 0.55rem;
  background: rgba(4, 8, 15, 0.45);
}

.model-preview-inline {
  flex: 0 0 auto;
  width: min(360px, 42vw);
  padding: 0.4rem 0.5rem;
}

.model-picked-preview p {
  margin: 0;
  color: var(--sub);
  font-size: 0.9rem;
}

.model-picked-card {
  display: flex;
  gap: 0.65rem;
  align-items: center;
}

.model-preview-inline .model-picked-card {
  gap: 0.5rem;
}

.model-picked-thumb {
  width: 120px;
  height: 68px;
  object-fit: cover;
  border-radius: 8px;
  border: 1px solid rgba(130, 174, 208, 0.35);
  background: rgba(4, 8, 15, 0.8);
}

.model-picked-empty {
  width: 120px;
  min-height: 68px;
  display: flex;
  align-items: center;
  justify-content: center;
  text-align: center;
  border-radius: 8px;
  border: 1px dashed rgba(130, 174, 208, 0.35);
  color: var(--sub);
  font-size: 0.8rem;
  padding: 0.3rem;
}

.model-preview-inline .model-picked-thumb,
.model-preview-inline .model-picked-empty {
  width: 92px;
  height: 52px;
  min-height: 52px;
}

.model-picked-meta {
  display: flex;
  flex-direction: column;
  gap: 0.2rem;
  min-width: 0;
}

.model-picked-meta a {
  color: #b6e7ff;
  text-decoration: none;
}

.model-picked-meta a:hover {
  text-decoration: underline;
}

.model-picked-meta span {
  color: var(--sub);
  font-size: 0.82rem;
  font-family: "IBM Plex Mono", monospace;
}

.model-preview-inline .model-picked-meta span {
  display: none;
}

#taskStatus {
  font-family: "IBM Plex Mono", monospace;
  font-size: 0.85rem;
  color: #d2e4f2;
  margin-bottom: 0.7rem;
}

.task-progress-wrap {
  position: relative;
  width: 100%;
  height: 26px;
  border-radius: 10px;
  border: 1px solid rgba(130, 174, 208, 0.45);
  background: rgba(4, 8, 15, 0.85);
  overflow: hidden;
  margin-bottom: 0.7rem;
}

.task-progress-bar {
  position: absolute;
  inset: 0 auto 0 0;
  width: 0%;
  background: linear-gradient(90deg, rgba(20, 197, 175, 0.85), rgba(255, 154, 60, 0.8));
  transition: width 200ms linear;
}

.task-progress-label {
  position: absolute;
  inset: 0;
  display: flex;
  align-items: center;
  justify-content: center;
  padding: 0 0.5rem;
  color: #eef8ff;
  font-family: "IBM Plex Mono", monospace;
  font-size: 0.78rem;
  text-shadow: 0 1px 2px rgba(0, 0, 0, 0.8);
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}

#preview {
  width: 100%;
  border-radius: 10px;
  border: 1px solid rgba(20, 197, 175, 0.45);
  background: #000;
  display: none;
}

#imagePreview {
  width: 100%;
  border-radius: 10px;
  border: 1px solid rgba(20, 197, 175, 0.45);
  background: #000;
  display: none;
  margin-bottom: 0.7rem;
  object-fit: contain;
}

@media (max-width: 760px) {
  .shell {
    margin: 0.8rem auto;
    padding: 0.8rem;
  }
  .top {
    flex-direction: column;
  }
  .top-right {
    align-items: stretch;
    width: 100%;
  }
  .lang-picker {
    justify-content: space-between;
  }
  .model-select-actions {
    flex-wrap: wrap;
    align-items: stretch;
  }
  .model-select-actions select {
    min-width: 0;
  }
  .model-preview-inline {
    width: 100%;
  }
  .model-search-query-row {
    grid-template-columns: 1fr;
  }
  .model-browser {
    grid-template-columns: 1fr;
  }
  .model-filter-pane {
    position: static;
  }
  .model-card.list-mode {
    grid-template-columns: 1fr;
  }
  .model-card.list-mode .model-card-cover {
    border-right: none;
    border-bottom: 1px solid rgba(130, 174, 208, 0.35);
  }
}

===== FILE: static/app.js =====

const SUPPORTED_LANGS = ["en", "ja", "es", "fr", "de", "it", "pt", "ru", "ar"];
const DEFAULT_LANG = "en";
const LANG_STORAGE_KEY = "videogen_lang";
const TASK_STORAGE_KEY = "videogen_last_task_id";
const TASK_POLL_INTERVAL_MS = 1000;

const I18N = {
  en: {
    appTitle: "ROCm VideoGen Studio",
    appSubtitle: "Text-to-Image / Image-to-Image / Text-to-Video / Image-to-Video",
    languageLabel: "Language",
    runtimeLoading: "runtime: loading...",
    tabTextToImage: "Text to Image",
    tabImageToImage: "Image to Image",
    tabTextToVideo: "Text to Video",
    tabImageToVideo: "Image to Video",
    tabModels: "Model Search",
    tabLocalModels: "Local Models",
    tabOutputs: "Outputs",
    tabSettings: "Settings",
    labelPrompt: "Prompt",
    labelNegativePrompt: "Negative Prompt",
    labelModelOptional: "Model ID (optional)",
    labelModelSelect: "Model Selection",
    labelSteps: "Steps",
    labelFrames: "Frames",
    labelGuidance: "Guidance",
    labelFps: "FPS",
    labelT2VBackend: "T2V Backend",
    labelSeed: "Seed",
    labelInputImage: "Input Image",
    labelWidth: "Width",
    labelHeight: "Height",
    labelLoraSelect: "LoRA Selection",
    labelVaeSelect: "VAE Selection",
    labelLoraScale: "LoRA Scale",
    labelStrength: "Strength",
    btnGenerateTextVideo: "Generate Text Video",
    btnGenerateImageVideo: "Generate Image Video",
    btnGenerateImageImage: "Generate Image Image",
    btnRefreshModels: "Refresh",
    btnRefreshLoras: "Refresh LoRAs",
    btnRefreshVaes: "Refresh VAEs",
    labelDownloadSavePathOptional: "Download Save Path (optional)",
    btnBrowsePath: "Browse",
    labelTask: "Task",
    labelSearchSource: "Source",
    labelSearchBaseModel: "Base Model",
    labelSearchSort: "Sort",
    labelSearchNsfw: "NSFW",
    labelSearchModelKind: "Model Kind",
    labelSearchViewMode: "View",
    labelQuery: "Query",
    labelLimit: "Limit",
    btnSearchModels: "Search Models",
    headingLocalModels: "Local Models",
    headingOutputs: "Outputs",
    btnRefreshLocalList: "Refresh Local List",
    btnRefreshOutputs: "Refresh Outputs",
    labelLocalModelsPath: "Local Models Path",
    labelLocalLineage: "Lineage",
    labelModelsDirectory: "Models Directory",
    labelListenPort: "Listen Port",
    labelRocmAotriton: "ROCm AOTriton Experimental",
    labelOutputsDirectory: "Outputs Directory",
    labelTempDirectory: "Temp Directory",
    labelLogLevel: "Log Level",
    labelHfToken: "HF Token",
    labelDefaultT2VBackend: "Default T2V Backend",
    labelT2VNpuRunner: "T2V NPU Runner",
    labelT2VNpuModelDir: "T2V NPU Model Directory",
    labelDefaultTextModel: "Default: Text to Video",
    labelDefaultImageModel: "Default: Image to Video",
    labelDefaultTextImageModel: "Default: Text to Image",
    labelDefaultImageImageModel: "Default: Image to Image",
    labelDefaultSteps: "Default Steps",
    labelDefaultFrames: "Default Frames",
    labelDefaultGuidance: "Default Guidance",
    labelDefaultFps: "Default FPS",
    labelDefaultWidth: "Default Width",
    labelDefaultHeight: "Default Height",
    labelClearHfCache: "Clear Hugging Face Cache",
    helpLanguage:
      "Impact: Changes UI language for labels/messages; generation behavior itself is unchanged.\nExample: English",
    helpPromptText2Video:
      "Impact: Main instruction that drives scene, motion, and style in text-to-video.\nExample: A cinematic drone shot over a neon city at night",
    helpPromptText2Image:
      "Impact: Main instruction that drives composition, subject, and style in text-to-image.\nExample: A photorealistic portrait of a fox in a tailored suit",
    helpNegativePromptText2Image:
      "Impact: Suppresses unwanted artifacts or styles in generated images.\nExample: blurry, low quality, watermark",
    helpLoraSelectText2Image:
      "Impact: Applies LoRA adapters to alter style/subject behavior for this model. Multiple selection is supported.\nExample: your-org/anime-style-lora",
    helpVaeSelectText2Image:
      "Impact: Replaces VAE for text-to-image quality and color/contrast characteristics.\nExample: stabilityai/sd-vae-ft-mse",
    helpModelSelectText2Image:
      "Impact: Changes the text-to-image model and affects style, quality, speed, and VRAM usage.\nExample: runwayml/stable-diffusion-v1-5",
    helpStepsText2Image:
      "Impact: Higher values may improve detail but increase generation time.\nExample: 30",
    helpGuidanceText2Image:
      "Impact: Higher guidance follows prompt more strongly but can reduce natural variation.\nExample: 7.5",
    helpWidthText2Image:
      "Impact: Higher width increases resolution/detail but uses more VRAM.\nExample: 512",
    helpHeightText2Image:
      "Impact: Higher height increases resolution/detail but uses more VRAM.\nExample: 512",
    helpSeedText2Image:
      "Impact: Fixes randomness for reproducible images; blank uses random seed.\nExample: 12345",
    helpPromptImage2Image:
      "Impact: Controls how the input image should be transformed while preserving structure.\nExample: Convert this photo into a cinematic film poster",
    helpNegativePromptImage2Image:
      "Impact: Reduces unwanted artifacts/styles during image-to-image generation.\nExample: blurry, distorted, watermark",
    helpModelSelectImage2Image:
      "Impact: Changes image-to-image model; style quality, speed, and VRAM usage can vary.\nExample: runwayml/stable-diffusion-v1-5",
    helpLoraSelectImage2Image:
      "Impact: Applies LoRA style/subject adaptation to image-to-image output. Multiple selection is supported.\nExample: your-org/lineart-lora",
    helpVaeSelectImage2Image:
      "Impact: Replaces VAE for image-to-image decode quality and color profile.\nExample: stabilityai/sd-vae-ft-mse",
    helpStepsImage2Image:
      "Impact: Higher values may improve detail but increase generation time.\nExample: 30",
    helpGuidanceImage2Image:
      "Impact: Higher guidance follows prompt more strongly but may reduce natural variation.\nExample: 7.5",
    helpStrengthImage2Image:
      "Impact: Controls how much the result changes from the input image.\nExample: 0.8",
    helpWidthImage2Image:
      "Impact: Higher width increases output detail but also VRAM use.\nExample: 512",
    helpHeightImage2Image:
      "Impact: Higher height increases output detail but also VRAM use.\nExample: 512",
    helpSeedImage2Image:
      "Impact: Fixes randomness for reproducible outputs; blank uses random seed.\nExample: 12345",
    helpNegativePromptText2Video:
      "Impact: Suppresses unwanted artifacts or styles during generation.\nExample: blurry, low quality, watermark",
    helpModelSelectText2Video:
      "Impact: Changes the text-to-video model; quality, speed, VRAM usage, and style can vary.\nExample: damo-vilab/text-to-video-ms-1.7b",
    helpLoraSelectText2Video:
      "Impact: Applies LoRA adapters for text-to-video generation when compatible. Multiple selection is supported.\nExample: your-org/cinematic-look-lora",
    helpStepsText2Video:
      "Impact: Higher values usually improve quality but increase time and GPU load.\nExample: 30",
    helpFramesText2Video:
      "Impact: More frames create longer/smoother clips but raise VRAM and render time.\nExample: 16",
    helpGuidanceText2Video:
      "Impact: Higher guidance follows prompt more strongly but may look less natural.\nExample: 9.0",
    helpFpsText2Video:
      "Impact: Controls output playback smoothness/speed.\nExample: 8",
    helpSeedText2Video:
      "Impact: Fixes randomness for reproducible outputs; blank uses random seed.\nExample: 12345",
    helpInputImage:
      "Impact: Source image defines starting composition and appearance for image-to-video.\nExample: C:\\Images\\input.png",
    helpPromptImage2Video:
      "Impact: Adds motion/style instructions on top of the input image.\nExample: Smooth cinematic camera pan with subtle parallax",
    helpNegativePromptImage2Video:
      "Impact: Reduces unwanted artifacts/flicker in generated video.\nExample: flicker, artifact, noisy",
    helpModelSelectImage2Video:
      "Impact: Changes the image-to-video model and affects motion quality, speed, and VRAM use.\nExample: ali-vilab/i2vgen-xl",
    helpLoraSelectImage2Video:
      "Impact: Applies LoRA adapters for image-to-video generation when compatible. Multiple selection is supported.\nExample: your-org/motion-style-lora",
    helpStepsImage2Video:
      "Impact: Higher values can improve quality but increase generation time.\nExample: 30",
    helpFramesImage2Video:
      "Impact: More frames increase clip duration and memory cost.\nExample: 16",
    helpGuidanceImage2Video:
      "Impact: Higher values increase prompt adherence; too high may reduce naturalness.\nExample: 9.0",
    helpFpsImage2Video:
      "Impact: Sets playback smoothness/speed for encoded video.\nExample: 8",
    helpWidthImage2Video:
      "Impact: Increases output resolution/detail and also VRAM usage.\nExample: 512",
    helpHeightImage2Video:
      "Impact: Increases output resolution/detail and also VRAM usage.\nExample: 512",
    helpSeedImage2Video:
      "Impact: Fixes randomness for reproducibility; blank gives random result each run.\nExample: 12345",
    helpDownloadSavePath:
      "Impact: Sets where downloaded models are stored for this operation.\nExample: D:\\ModelStore\\VideoGen",
    helpTask:
      "Impact: Filters model search target by generation type.\nExample: text-to-video",
    helpSearchSource:
      "Impact: Selects which service to query for models; CivitAI is available for image tasks.\nExample: all",
    helpSearchBaseModel:
      "Impact: Filters search results by base model family; helps narrow compatible checkpoints quickly.\nExample: StableDiffusion XL",
    helpQuery:
      "Impact: Narrows search results to models matching keywords; empty shows popular models.\nExample: i2vgen",
    helpLimit:
      "Impact: Controls number of search results shown; larger values increase list size and fetch time.\nExample: 30",
    helpSearchSort:
      "Impact: Changes ranking of search results by metric/time.\nExample: downloads",
    helpSearchNsfw:
      "Impact: Includes or excludes NSFW models from provider search (provider support dependent).\nExample: exclude",
    helpSearchModelKind:
      "Impact: Filters model type such as checkpoint/LoRA/VAE when provider supports it.\nExample: checkpoint",
    helpSearchViewMode:
      "Impact: Changes only the card layout (grid/list), not search results.\nExample: grid",
    helpLocalModelsPath:
      "Impact: Changes which folder is scanned and shown in the Local Models screen.\nExample: D:\\ModelStore\\VideoGen",
    helpLocalLineage:
      "Impact: Filters local model list by model family lineage.\nExample: StableDiffusion XL",
    helpModelsDirectory:
      "Impact: Changes where downloaded models are stored and where local model scanning occurs.\nExample: C:\\AI\\VideoGen\\models",
    helpListenPort:
      "Impact: Changes server listening port used at next startup; restart is required after save.\nExample: 8000",
    helpRocmAotriton:
      "Impact: Enables/disables ROCm experimental AOTriton SDPA/Flash attention path at next startup. ON can improve step speed, OFF may improve stability on some environments.\nExample: 1=enabled, 0=disabled",
    helpOutputsDirectory:
      "Impact: Changes where generated video files are written.\nExample: D:\\VideoOutputs",
    helpTempDirectory:
      "Impact: Changes where temporary upload/intermediate files are stored during processing.\nExample: C:\\AI\\VideoGen\\tmp",
    helpLogLevel:
      "Impact: Controls log verbosity for troubleshooting. DEBUG outputs detailed internal steps and may increase log size.\nExample: DEBUG",
    helpHfToken:
      "Impact: Enables access to private/gated Hugging Face models and can improve API limits.\nExample: hf_xxxxxxxxxxxxxxxxxxxxx",
    helpT2VBackend:
      "Impact: Selects execution backend for this Text-to-Video request. auto uses settings/default routing; NPU requires a configured runner.\nExample: npu",
    helpDefaultT2VBackend:
      "Impact: Sets default Text-to-Video backend when form backend is auto.\nExample: auto",
    helpT2VNpuRunner:
      "Impact: Sets executable/script path for NPU Text-to-Video runner. The runner must accept '--input-json <path>' and create output video.\nExample: C:\\AI\\NPU\\t2v_runner.bat",
    helpT2VNpuModelDir:
      "Impact: Optional model directory passed to NPU runner for ONNX/NPU models.\nExample: C:\\AI\\VideoGen\\models_npu\\text2video",
    logLevelDebug: "DEBUG",
    logLevelInfo: "INFO",
    logLevelWarning: "WARNING",
    logLevelError: "ERROR",
    helpDefaultTextModel:
      "Impact: Sets the model preselected/used for text-to-video when no explicit model is selected.\nExample: damo-vilab/text-to-video-ms-1.7b",
    helpDefaultImageModel:
      "Impact: Sets the model preselected/used for image-to-video when no explicit model is selected.\nExample: ali-vilab/i2vgen-xl",
    helpDefaultTextImageModel:
      "Impact: Sets the model preselected/used for text-to-image when no explicit model is selected.\nExample: runwayml/stable-diffusion-v1-5",
    helpDefaultImageImageModel:
      "Impact: Sets the model preselected/used for image-to-image when no explicit model is selected.\nExample: runwayml/stable-diffusion-v1-5",
    helpLoraScale:
      "Impact: Controls LoRA effect strength; larger values emphasize LoRA style/features more. This value is applied to all selected LoRAs.\nExample: 1.0",
    helpDefaultSteps:
      "Impact: Higher values improve quality but increase generation time and GPU load.\nExample: 30",
    helpDefaultFrames:
      "Impact: Higher frame count makes longer/smoother clips but increases VRAM and processing time.\nExample: 16",
    helpDefaultGuidance:
      "Impact: Higher guidance follows prompt more strongly but may reduce natural motion.\nExample: 9.0",
    helpDefaultFps:
      "Impact: Controls playback speed and smoothness of output video.\nExample: 8",
    helpDefaultWidth:
      "Impact: Higher resolution improves detail but significantly increases VRAM usage.\nExample: 512",
    helpDefaultHeight:
      "Impact: Higher resolution improves detail but significantly increases VRAM usage.\nExample: 512",
    helpClearHfCache:
      "Impact: Deletes downloaded Hugging Face cache files to reclaim disk space; next model load/download may take longer.\nExample: Run this after low disk warning",
    btnSaveSettings: "Save Settings",
    btnClearHfCache: "Clear Cache",
    btnGenerateTextImage: "Generate Text Image",
    headingPathBrowser: "Folder Browser",
    btnClosePathBrowser: "Close",
    labelCurrentPath: "Current Path",
    btnRoots: "Roots",
    btnUpFolder: "Up",
    btnUseThisPath: "Use This Path",
    statusNoTask: "No task running.",
    placeholderT2IPrompt: "A portrait photo of a fox wearing a suit, studio light",
    placeholderI2IPrompt: "Refine this image into a cinematic poster style",
    placeholderT2VPrompt: "A cinematic drone shot above neon city...",
    placeholderNegativePrompt: "low quality, blurry",
    placeholderI2VPrompt: "Turn this image into a smooth cinematic motion...",
    placeholderI2VNegativePrompt: "artifact, flicker",
    placeholderI2INegativePrompt: "blurry, low quality",
    placeholderSeed: "random if empty",
    placeholderDownloadSavePath: "empty = use Models Directory from Settings",
    placeholderSearchQuery: "i2vgen, text-to-video...",
    placeholderLocalModelsPath: "empty = use Models Directory from Settings",
    placeholderOptional: "optional",
    searchSourceAll: "All",
    searchSourceHf: "Hugging Face",
    searchSourceCivitai: "CivitAI",
    searchBaseModelAll: "All base models",
    localLineageAll: "All lineages",
    msgSettingsSaved: "Settings saved.",
    msgNoLocalModels: "No local models in: {path}",
    msgNoOutputs: "No outputs in: {path}",
    msgPortChangeSaved: "Listen port saved. Restart `start.bat` to apply new port.",
    msgServerSettingRestartRequired: "Server setting saved. Restart `start.bat` to apply changes.",
    msgNoModelsFound: "No models found.",
    msgModelDetailEmpty: "Select a model to view detail.",
    msgModelDetailLoading: "Loading model detail...",
    msgModelDetailLoadFailed: "Model detail failed: {error}",
    msgModelInstalled: "Downloaded",
    msgModelNotInstalled: "Not downloaded",
    msgSearchPage: "Page {page}",
    msgApply: "Apply",
    msgDetail: "Detail",
    msgDetailDescription: "Description",
    msgDetailTags: "Tags",
    msgDetailVersions: "Version",
    msgDetailFiles: "File",
    msgDetailRevision: "Revision",
    msgSearchModelApplied: "Model set for {task}: {model}",
    msgDefaultModelOption: "Use default model ({model})",
    msgDefaultModelNoMeta: "Use default model",
    msgNoModelCatalog: "No models available.",
    msgNoLoraCatalog: "No LoRAs available for this model.",
    msgNoLoraOption: "No LoRA",
    msgNoVaeOption: "No VAE",
    msgSearchLineage: "Search lineage",
    msgSetTaskModel: "Set {task}",
    msgLocalModelApplied: "Local model set for {task}: {model}",
    msgLineageSearchStarted: "Searching lineage from base model: {base}",
    msgModelSelectHint: "Select a model to see thumbnail.",
    msgModelNoPreview: "No thumbnail available for this model.",
    msgNoFolders: "No subfolders found.",
    msgOpen: "Open",
    btnDownload: "Download",
    btnPrev: "Prev",
    btnNext: "Next",
    btnDeleteModel: "Delete",
    msgAlreadyDownloaded: "Downloaded",
    msgModelPreviewAlt: "Model preview",
    msgModelDownloadStarted: "Model download started: {repo} -> {path}",
    msgTextImageGenerationStarted: "Text-to-image generation started: {id}",
    msgImageImageGenerationStarted: "Image-to-image generation started: {id}",
    msgTextGenerationStarted: "Text generation started: {id}",
    msgImageGenerationStarted: "Image generation started: {id}",
    msgTaskPollFailed: "Task poll failed: {error}",
    msgConfirmDeleteModel: "Delete local model '{model}'?",
    msgModelDeleted: "Model deleted: {model}",
    msgModelDeleteFailed: "Model delete failed: {error}",
    msgConfirmDeleteOutput: "Delete output '{name}'?",
    msgOutputDeleted: "Output deleted: {name}",
    msgOutputDeleteFailed: "Output delete failed: {error}",
    msgOutputsRefreshFailed: "Outputs refresh failed: {error}",
    msgConfirmClearHfCache: "Delete Hugging Face cache now? Cached files will be re-downloaded later.",
    msgHfCacheCleared: "Hugging Face cache cleared. removed={removed}, skipped={skipped}, failed={failed}",
    msgHfCacheClearFailed: "Hugging Face cache clear failed: {error}",
    msgSaveSettingsFailed: "Save settings failed: {error}",
    msgSearchFailed: "Search failed: {error}",
    msgTextGenerationFailed: "Text generation failed: {error}",
    msgImageGenerationFailed: "Image generation failed: {error}",
    msgLocalModelRefreshFailed: "Local model refresh failed: {error}",
    msgPathBrowserLoadFailed: "Folder browser load failed: {error}",
    msgInitFailed: "Initialization failed: {error}",
    msgInputImageRequired: "Input image is required.",
    msgDefaultModelsDir: "default models dir",
    msgSelectLocalModel: "Select local model",
    msgDefaultModelNotLocal: "(not local) {model}",
    msgUnknownPath: "(unknown)",
    modelTag: "tag",
    outputUpdated: "updated",
    outputTypeImage: "image",
    outputTypeVideo: "video",
    outputTypeOther: "other",
    modelKind: "kind",
    modelKindBase: "Base",
    modelKindLora: "LoRA",
    modelKindVae: "VAE",
    modelBase: "base",
    modelDownloads: "downloads",
    modelLikes: "likes",
    modelSize: "size",
    modelSource: "source",
    taskTypeText2Image: "text2image",
    taskTypeImage2Image: "image2image",
    taskTypeText2Video: "text2video",
    taskTypeImage2Video: "image2video",
    taskTypeDownload: "download",
    taskTypeUnknown: "unknown",
    statusQueued: "queued",
    statusRunning: "running",
    statusCompleted: "completed",
    statusError: "error",
    taskLine: "task={id} | type={type} | status={status} | {progress}% | {message}",
    taskError: "error={error}",
    runtimeDevice: "device",
    runtimeCuda: "cuda",
    runtimeRocm: "rocm",
    runtimeNpu: "npu",
    runtimeNpuReason: "npu_reason",
    runtimeDiffusers: "diffusers",
    runtimeTorch: "torch",
    runtimeError: "error",
    backendAuto: "Auto",
    backendCuda: "GPU (CUDA/ROCm)",
    backendNpu: "NPU",
    runtimeLoadFailed: "runtime load failed: {error}",
    serverQueued: "Queued",
    serverGenerationQueued: "Generation queued",
    serverDownloadQueued: "Download queued",
    serverLoadingModel: "Loading model",
    serverLoadingLora: "Applying LoRA",
    serverPreparingImage: "Preparing image",
    serverGeneratingImage: "Generating image",
    serverGeneratingFrames: "Generating frames",
    serverDecodingLatents: "Decoding latents",
    serverDecodingLatentsCpuFallback: "Decoding latents",
    serverPostprocessingImage: "Postprocessing image",
    serverEncoding: "Encoding mp4",
    serverSavingPng: "Saving png",
    serverDone: "Done",
    serverGenerationFailed: "Generation failed",
    serverDownloadComplete: "Download complete",
    serverDownloadFailed: "Download failed",
  },
  ja: {
    appTitle: "ROCm VideoGen Studio",
    appSubtitle: "テキスト画像生成 / 画像画像生成 / テキスト動画生成 / 画像動画生成",
    languageLabel: "言語",
    runtimeLoading: "実行環境: 読み込み中...",
    tabTextToImage: "テキスト→画像",
    tabImageToImage: "画像→画像",
    tabTextToVideo: "テキスト→動画",
    tabImageToVideo: "画像→動画",
    tabModels: "モデル検索",
    tabLocalModels: "ローカルモデル",
    tabOutputs: "成果物",
    tabSettings: "設定",
    labelPrompt: "プロンプト",
    labelNegativePrompt: "ネガティブプロンプト",
    labelModelOptional: "モデルID（任意）",
    labelModelSelect: "モデル選択",
    labelSteps: "ステップ数",
    labelFrames: "フレーム数",
    labelGuidance: "ガイダンス",
    labelFps: "FPS",
    labelT2VBackend: "T2Vバックエンド",
    labelSeed: "シード",
    labelInputImage: "入力画像",
    labelWidth: "幅",
    labelHeight: "高さ",
    labelLoraSelect: "LoRA選択",
    labelVaeSelect: "VAE選択",
    labelLoraScale: "LoRA強度",
    labelStrength: "変換強度",
    btnGenerateTextVideo: "テキスト動画を生成",
    btnGenerateImageVideo: "画像動画を生成",
    btnGenerateImageImage: "画像画像を生成",
    btnRefreshModels: "更新",
    btnRefreshLoras: "LoRA更新",
    btnRefreshVaes: "VAE更新",
    labelDownloadSavePathOptional: "モデル保存先（任意）",
    btnBrowsePath: "参照",
    labelTask: "タスク",
    labelSearchSource: "検索元",
    labelSearchBaseModel: "ベースモデル",
    labelSearchSort: "並び替え",
    labelSearchNsfw: "NSFW",
    labelSearchModelKind: "モデル種別",
    labelSearchViewMode: "表示",
    labelQuery: "検索語",
    labelLimit: "件数",
    btnSearchModels: "モデル検索",
    headingLocalModels: "ローカルモデル",
    headingOutputs: "成果物",
    btnRefreshLocalList: "ローカル一覧を更新",
    btnRefreshOutputs: "成果物一覧を更新",
    labelLocalModelsPath: "ローカルモデル表示パス",
    labelLocalLineage: "系譜",
    labelModelsDirectory: "モデル保存ディレクトリ",
    labelListenPort: "リッスンポート",
    labelRocmAotriton: "ROCm AOTriton 実験機能",
    labelOutputsDirectory: "出力ディレクトリ",
    labelTempDirectory: "一時ディレクトリ",
    labelLogLevel: "ログレベル",
    labelHfToken: "HFトークン",
    labelDefaultT2VBackend: "既定T2Vバックエンド",
    labelT2VNpuRunner: "T2V NPUランナー",
    labelT2VNpuModelDir: "T2V NPUモデルディレクトリ",
    labelDefaultTextModel: "既定: テキスト→動画",
    labelDefaultImageModel: "既定: 画像→動画",
    labelDefaultTextImageModel: "既定: テキスト→画像",
    labelDefaultImageImageModel: "既定: 画像→画像",
    labelDefaultSteps: "既定ステップ数",
    labelDefaultFrames: "既定フレーム数",
    labelDefaultGuidance: "既定ガイダンス",
    labelDefaultFps: "既定FPS",
    labelDefaultWidth: "既定幅",
    labelDefaultHeight: "既定高さ",
    labelClearHfCache: "Hugging Face キャッシュ削除",
    helpLanguage:
      "影響: 画面の表示言語が変わります。生成ロジック自体には影響しません。\n例: 日本語",
    helpPromptText2Video:
      "影響: テキスト→動画の内容・動き・雰囲気を決める主指示です。\n例: 夜のネオン都市を俯瞰するシネマティックなドローン映像",
    helpPromptText2Image:
      "影響: テキスト→画像の構図・被写体・画風を決める主指示です。\n例: スーツを着たキツネの写実的なポートレート",
    helpNegativePromptText2Image:
      "影響: 画像で避けたい要素や品質低下要因を抑制します。\n例: blurry, low quality, watermark",
    helpLoraSelectText2Image:
      "影響: このモデルにLoRAを適用し、画風や被写体特性を調整できます。複数選択に対応します。\n例: your-org/anime-style-lora",
    helpVaeSelectText2Image:
      "影響: テキスト→画像のVAEを差し替え、発色や復元品質に影響します。\n例: stabilityai/sd-vae-ft-mse",
    helpModelSelectText2Image:
      "影響: 使用モデルにより画風・品質・速度・VRAM使用量が変わります。\n例: runwayml/stable-diffusion-v1-5",
    helpStepsText2Image:
      "影響: 値を上げると細部品質向上が見込めますが、時間が増えます。\n例: 30",
    helpGuidanceText2Image:
      "影響: 値を上げると指示忠実度が上がりますが、自然な揺らぎが減る場合があります。\n例: 7.5",
    helpWidthText2Image:
      "影響: 値を上げると解像度/細部が向上しますがVRAM使用量が増えます。\n例: 512",
    helpHeightText2Image:
      "影響: 値を上げると解像度/細部が向上しますがVRAM使用量が増えます。\n例: 512",
    helpSeedText2Image:
      "影響: 乱数を固定して再現しやすくなります。空欄は毎回ランダムです。\n例: 12345",
    helpPromptImage2Image:
      "影響: 入力画像をどの方向に変換するかを指定します。\n例: この写真を映画ポスター風に変換",
    helpNegativePromptImage2Image:
      "影響: 画像変換で不要な要素や劣化を抑制します。\n例: blurry, distorted, watermark",
    helpModelSelectImage2Image:
      "影響: 画像→画像モデルにより画風・品質・速度・VRAM使用量が変わります。\n例: runwayml/stable-diffusion-v1-5",
    helpLoraSelectImage2Image:
      "影響: 画像→画像にLoRAを適用してスタイルや特徴を強調します。複数選択に対応します。\n例: your-org/lineart-lora",
    helpVaeSelectImage2Image:
      "影響: 画像→画像のVAEを差し替え、発色や復元品質に影響します。\n例: stabilityai/sd-vae-ft-mse",
    helpStepsImage2Image:
      "影響: 値を上げると細部品質が向上しやすい一方、処理時間が増えます。\n例: 30",
    helpGuidanceImage2Image:
      "影響: 値を上げるとプロンプト忠実度が上がります。\n例: 7.5",
    helpStrengthImage2Image:
      "影響: 入力画像からどれだけ変化させるかを調整します。\n例: 0.8",
    helpWidthImage2Image:
      "影響: 値を上げると精細になりますがVRAM使用量が増えます。\n例: 512",
    helpHeightImage2Image:
      "影響: 値を上げると精細になりますがVRAM使用量が増えます。\n例: 512",
    helpSeedImage2Image:
      "影響: 乱数を固定し再現しやすくなります。空欄は毎回ランダムです。\n例: 12345",
    helpNegativePromptText2Video:
      "影響: 出したくない要素や品質低下要因を抑制します。\n例: blurry, low quality, watermark",
    helpModelSelectText2Video:
      "影響: 使用モデルにより品質・速度・VRAM使用量・画風が変わります。\n例: damo-vilab/text-to-video-ms-1.7b",
    helpLoraSelectText2Video:
      "影響: 互換モデルの場合、LoRAで画風や特徴を追加できます。複数選択に対応します。\n例: your-org/cinematic-look-lora",
    helpStepsText2Video:
      "影響: 値を上げると品質向上が見込めますが、時間とGPU負荷が増えます。\n例: 30",
    helpFramesText2Video:
      "影響: 値を上げると動画が長く滑らかになりますが、VRAMと処理時間が増えます。\n例: 16",
    helpGuidanceText2Video:
      "影響: 値を上げると指示への忠実度が上がりますが、不自然になる場合があります。\n例: 9.0",
    helpFpsText2Video:
      "影響: 出力動画の再生速度・滑らかさに影響します。\n例: 8",
    helpSeedText2Video:
      "影響: 乱数を固定し、同条件で再現しやすくなります。空欄は毎回ランダムです。\n例: 12345",
    helpInputImage:
      "影響: 画像→動画の構図や見た目の基準画像になります。\n例: C:\\Images\\input.png",
    helpPromptImage2Video:
      "影響: 入力画像に対する動きや演出の指示を与えます。\n例: 滑らかなカメラパンと弱いパララックス",
    helpNegativePromptImage2Video:
      "影響: ちらつきやノイズなど不要な要素を抑制します。\n例: flicker, artifact, noisy",
    helpModelSelectImage2Video:
      "影響: 使用モデルで動き品質・速度・VRAM使用量が変化します。\n例: ali-vilab/i2vgen-xl",
    helpLoraSelectImage2Video:
      "影響: 互換モデルの場合、LoRAで画風や特徴を追加できます。複数選択に対応します。\n例: your-org/motion-style-lora",
    helpStepsImage2Video:
      "影響: 値を上げると品質向上が見込めますが処理時間が増えます。\n例: 30",
    helpFramesImage2Video:
      "影響: 値を上げると動画長が伸びますがメモリ消費も増えます。\n例: 16",
    helpGuidanceImage2Video:
      "影響: 値を上げると指示忠実度が上がりますが自然さが下がる場合があります。\n例: 9.0",
    helpFpsImage2Video:
      "影響: 出力動画の滑らかさ・再生速度に影響します。\n例: 8",
    helpWidthImage2Video:
      "影響: 解像度が上がり精細になりますがVRAM使用量が増えます。\n例: 512",
    helpHeightImage2Video:
      "影響: 解像度が上がり精細になりますがVRAM使用量が増えます。\n例: 512",
    helpSeedImage2Video:
      "影響: 乱数固定により再現しやすくなります。空欄は毎回ランダムです。\n例: 12345",
    helpDownloadSavePath:
      "影響: この操作でダウンロードするモデルの保存先が変わります。\n例: D:\\ModelStore\\VideoGen",
    helpTask:
      "影響: モデル検索対象を生成タイプで絞り込みます。\n例: text-to-video",
    helpSearchSource:
      "影響: モデル検索の参照先を選びます。CivitAIは画像系タスクで利用できます。\n例: all",
    helpSearchBaseModel:
      "影響: ベースモデル系譜で検索結果を絞り込み、互換候補を見つけやすくします。\n例: StableDiffusion XL",
    helpQuery:
      "影響: キーワード一致で検索結果を絞ります。空欄では人気モデルを表示します。\n例: i2vgen",
    helpLimit:
      "影響: 表示件数を調整します。値が大きいほど表示数が増え、取得時間も増える場合があります。\n例: 30",
    helpSearchSort:
      "影響: 検索結果の並び順（DL数・いいね・更新時刻など）を変更します。\n例: downloads",
    helpSearchNsfw:
      "影響: NSFWモデルを含める/除外する指定です（提供元対応に依存）。\n例: exclude",
    helpSearchModelKind:
      "影響: checkpoint/LoRA/VAEなどモデル種別で絞り込みます（提供元対応に依存）。\n例: checkpoint",
    helpSearchViewMode:
      "影響: 結果の表示レイアウトのみを切り替えます（検索結果自体は変わりません）。\n例: grid",
    helpLocalModelsPath:
      "影響: ローカルモデル画面で一覧表示するフォルダを切り替えます。\n例: D:\\ModelStore\\VideoGen",
    helpLocalLineage:
      "影響: ローカルモデル一覧をモデル系譜で絞り込みます。\n例: StableDiffusion XL",
    helpModelsDirectory:
      "影響: ダウンロードモデルの保存先とローカルモデル検索先が変わります。\n例: C:\\AI\\VideoGen\\models",
    helpListenPort:
      "影響: 次回起動時のサーバー待受ポートが変わります。保存後に再起動が必要です。\n例: 8000",
    helpRocmAotriton:
      "影響: 次回起動時の ROCm 実験的 AOTriton（SDPA/Flash attention）を有効/無効化します。ONでSTEP速度が上がる場合があり、OFFで安定する場合があります。\n例: 1=有効, 0=無効",
    helpOutputsDirectory:
      "影響: 生成した動画ファイルの出力先が変わります。\n例: D:\\VideoOutputs",
    helpTempDirectory:
      "影響: アップロード画像や中間ファイルの一時保存先が変わります。\n例: C:\\AI\\VideoGen\\tmp",
    helpLogLevel:
      "影響: トラブル調査向けのログ詳細度を変更します。DEBUGは内部状態を詳細出力し、ログ量が増えます。\n例: DEBUG",
    helpHfToken:
      "影響: 非公開/制限付き Hugging Face モデルへのアクセスやAPI利用制限に影響します。\n例: hf_xxxxxxxxxxxxxxxxxxxxx",
    helpT2VBackend:
      "影響: このテキスト→動画リクエストの実行バックエンドを選択します。autoは設定値/自動判定に従います。NPUはランナー設定が必要です。\n例: npu",
    helpDefaultT2VBackend:
      "影響: テキスト→動画フォームでバックエンドがauto時に使う既定値を設定します。\n例: auto",
    helpT2VNpuRunner:
      "影響: NPU用テキスト→動画ランナーの実行ファイル/スクリプトパスを設定します。ランナーは '--input-json <path>' を受け取り出力動画を作成する必要があります。\n例: C:\\AI\\NPU\\t2v_runner.bat",
    helpT2VNpuModelDir:
      "影響: NPUランナー向けのモデルディレクトリを任意指定します（ONNX等）。未指定時は選択モデルの解決先を使用します。\n例: C:\\AI\\VideoGen\\models_npu\\text2video",
    logLevelDebug: "DEBUG",
    logLevelInfo: "INFO",
    logLevelWarning: "WARNING",
    logLevelError: "ERROR",
    helpDefaultTextModel:
      "影響: テキスト→動画でモデル未指定時に使われる既定モデルが変わります。\n例: damo-vilab/text-to-video-ms-1.7b",
    helpDefaultImageModel:
      "影響: 画像→動画でモデル未指定時に使われる既定モデルが変わります。\n例: ali-vilab/i2vgen-xl",
    helpDefaultTextImageModel:
      "影響: テキスト→画像でモデル未指定時に使われる既定モデルが変わります。\n例: runwayml/stable-diffusion-v1-5",
    helpDefaultImageImageModel:
      "影響: 画像→画像でモデル未指定時に使われる既定モデルが変わります。\n例: runwayml/stable-diffusion-v1-5",
    helpLoraScale:
      "影響: LoRAの効き具合を調整します。大きいほどLoRAの特徴が強く反映されます。複数選択時は全LoRAに同じ値を適用します。\n例: 1.0",
    helpDefaultSteps:
      "影響: 値を上げると品質向上が見込めますが、生成時間とGPU負荷が増えます。\n例: 30",
    helpDefaultFrames:
      "影響: 値を上げると動画が長く滑らかになりますが、VRAM消費と処理時間が増えます。\n例: 16",
    helpDefaultGuidance:
      "影響: 値を上げるとプロンプト忠実度が上がりますが、不自然な動きになる場合があります。\n例: 9.0",
    helpDefaultFps:
      "影響: 出力動画の再生速度・滑らかさに影響します。\n例: 8",
    helpDefaultWidth:
      "影響: 解像度が上がり精細になりますが、VRAM使用量が増えます。\n例: 512",
    helpDefaultHeight:
      "影響: 解像度が上がり精細になりますが、VRAM使用量が増えます。\n例: 512",
    helpClearHfCache:
      "影響: Hugging Face のキャッシュファイルを削除してディスク容量を回復します。次回は再ダウンロードが発生し時間がかかります。\n例: 容量不足警告が出た後に実行",
    btnSaveSettings: "設定を保存",
    btnClearHfCache: "キャッシュ削除",
    btnGenerateTextImage: "テキスト画像を生成",
    headingPathBrowser: "フォルダブラウザ",
    btnClosePathBrowser: "閉じる",
    labelCurrentPath: "現在のパス",
    btnRoots: "ルート",
    btnUpFolder: "上へ",
    btnUseThisPath: "このパスを使用",
    statusNoTask: "実行中のタスクはありません。",
    placeholderT2IPrompt: "スーツを着たキツネのスタジオポートレート...",
    placeholderI2IPrompt: "この画像を映画ポスター風に変換する...",
    placeholderT2VPrompt: "ネオン都市上空を飛ぶシネマティックなドローン映像...",
    placeholderNegativePrompt: "低品質, ぼやけ",
    placeholderI2VPrompt: "この画像に滑らかな映画的モーションを付ける...",
    placeholderI2VNegativePrompt: "artifact, flicker",
    placeholderI2INegativePrompt: "blur, low quality",
    placeholderSeed: "空欄でランダム",
    placeholderDownloadSavePath: "空欄 = 設定のモデル保存先を使用",
    placeholderSearchQuery: "i2vgen, text-to-video...",
    placeholderLocalModelsPath: "空欄 = 設定のモデル保存先を使用",
    placeholderOptional: "任意",
    searchSourceAll: "すべて",
    searchSourceHf: "Hugging Face",
    searchSourceCivitai: "CivitAI",
    searchBaseModelAll: "すべてのベースモデル",
    localLineageAll: "すべての系譜",
    msgSettingsSaved: "設定を保存しました。",
    msgNoLocalModels: "ローカルモデルがありません: {path}",
    msgNoOutputs: "成果物がありません: {path}",
    msgPortChangeSaved: "リッスンポートを保存しました。反映には `start.bat` の再起動が必要です。",
    msgServerSettingRestartRequired: "サーバー設定を保存しました。反映には `start.bat` の再起動が必要です。",
    msgNoModelsFound: "モデルが見つかりません。",
    msgModelDetailEmpty: "モデルを選択すると詳細を表示します。",
    msgModelDetailLoading: "モデル詳細を読み込み中...",
    msgModelDetailLoadFailed: "モデル詳細の取得に失敗: {error}",
    msgModelInstalled: "ダウンロード済み",
    msgModelNotInstalled: "未ダウンロード",
    msgSearchPage: "ページ {page}",
    msgApply: "適用",
    msgDetail: "詳細",
    msgDetailDescription: "説明",
    msgDetailTags: "タグ",
    msgDetailVersions: "バージョン",
    msgDetailFiles: "ファイル",
    msgDetailRevision: "リビジョン",
    msgSearchModelApplied: "{task} にモデルを設定: {model}",
    msgDefaultModelOption: "既定モデルを使用 ({model})",
    msgDefaultModelNoMeta: "既定モデルを使用",
    msgNoModelCatalog: "利用可能なモデルがありません。",
    msgNoLoraCatalog: "このモデルで利用可能なLoRAがありません。",
    msgNoLoraOption: "LoRAなし",
    msgNoVaeOption: "VAEなし",
    msgSearchLineage: "系譜検索",
    msgSetTaskModel: "{task}に設定",
    msgLocalModelApplied: "{task} にローカルモデルを設定: {model}",
    msgLineageSearchStarted: "ベースモデルから系譜を検索: {base}",
    msgModelSelectHint: "モデルを選択するとサムネイルを表示します。",
    msgModelNoPreview: "このモデルにはサムネイルがありません。",
    msgNoFolders: "サブフォルダがありません。",
    msgOpen: "開く",
    btnDownload: "ダウンロード",
    btnPrev: "前へ",
    btnNext: "次へ",
    btnDeleteModel: "削除",
    msgAlreadyDownloaded: "ダウンロード済み",
    msgModelPreviewAlt: "モデルプレビュー",
    msgModelDownloadStarted: "モデルダウンロード開始: {repo} -> {path}",
    msgTextImageGenerationStarted: "テキスト→画像生成を開始しました: {id}",
    msgImageImageGenerationStarted: "画像→画像生成を開始しました: {id}",
    msgTextGenerationStarted: "テキスト生成を開始しました: {id}",
    msgImageGenerationStarted: "画像生成を開始しました: {id}",
    msgTaskPollFailed: "タスク確認に失敗: {error}",
    msgConfirmDeleteModel: "ローカルモデル '{model}' を削除しますか？",
    msgModelDeleted: "モデルを削除しました: {model}",
    msgModelDeleteFailed: "モデル削除に失敗: {error}",
    msgConfirmDeleteOutput: "成果物 '{name}' を削除しますか？",
    msgOutputDeleted: "成果物を削除しました: {name}",
    msgOutputDeleteFailed: "成果物の削除に失敗: {error}",
    msgOutputsRefreshFailed: "成果物一覧の更新に失敗: {error}",
    msgConfirmClearHfCache: "Hugging Face キャッシュを削除しますか？削除後は必要なファイルが再ダウンロードされます。",
    msgHfCacheCleared: "Hugging Face キャッシュを削除しました。削除={removed}, スキップ={skipped}, 失敗={failed}",
    msgHfCacheClearFailed: "Hugging Face キャッシュ削除に失敗: {error}",
    msgSaveSettingsFailed: "設定保存に失敗: {error}",
    msgSearchFailed: "検索に失敗: {error}",
    msgTextGenerationFailed: "テキスト生成に失敗: {error}",
    msgImageGenerationFailed: "画像生成に失敗: {error}",
    msgLocalModelRefreshFailed: "ローカル一覧更新に失敗: {error}",
    msgPathBrowserLoadFailed: "フォルダブラウザの読み込み失敗: {error}",
    msgInitFailed: "初期化に失敗: {error}",
    msgInputImageRequired: "入力画像が必要です。",
    msgDefaultModelsDir: "既定のモデル保存先",
    msgSelectLocalModel: "ローカルモデルを選択",
    msgDefaultModelNotLocal: "（ローカル未配置）{model}",
    msgUnknownPath: "(不明)",
    modelTag: "タグ",
    outputUpdated: "更新日時",
    outputTypeImage: "画像",
    outputTypeVideo: "動画",
    outputTypeOther: "その他",
    modelKind: "種別",
    modelKindBase: "ベース",
    modelKindLora: "LoRA",
    modelKindVae: "VAE",
    modelBase: "ベースモデル",
    modelDownloads: "DL数",
    modelLikes: "いいね",
    modelSize: "サイズ",
    modelSource: "ソース",
    taskTypeText2Image: "テキスト画像",
    taskTypeImage2Image: "画像画像",
    taskTypeText2Video: "テキスト動画",
    taskTypeImage2Video: "画像動画",
    taskTypeDownload: "ダウンロード",
    taskTypeUnknown: "不明",
    statusQueued: "待機中",
    statusRunning: "実行中",
    statusCompleted: "完了",
    statusError: "エラー",
    taskLine: "task={id} | type={type} | status={status} | {progress}% | {message}",
    taskError: "error={error}",
    runtimeDevice: "device",
    runtimeCuda: "cuda",
    runtimeRocm: "rocm",
    runtimeNpu: "npu",
    runtimeNpuReason: "npu_reason",
    runtimeDiffusers: "diffusers",
    runtimeTorch: "torch",
    runtimeError: "error",
    backendAuto: "自動",
    backendCuda: "GPU (CUDA/ROCm)",
    backendNpu: "NPU",
    runtimeLoadFailed: "実行環境の取得に失敗: {error}",
    serverQueued: "キュー待ち",
    serverGenerationQueued: "生成キューに追加",
    serverDownloadQueued: "ダウンロードキューに追加",
    serverLoadingModel: "モデルを読み込み中",
    serverLoadingLora: "LoRAを適用中",
    serverPreparingImage: "画像を準備中",
    serverGeneratingImage: "画像を生成中",
    serverGeneratingFrames: "フレームを生成中",
    serverDecodingLatents: "潜在表現をデコード中",
    serverDecodingLatentsCpuFallback: "潜在表現をデコード中",
    serverPostprocessingImage: "画像を後処理中",
    serverEncoding: "mp4に変換中",
    serverSavingPng: "pngを保存中",
    serverDone: "完了",
    serverGenerationFailed: "生成に失敗",
    serverDownloadComplete: "ダウンロード完了",
    serverDownloadFailed: "ダウンロード失敗",
  },
  es: {
    languageLabel: "Idioma",
    tabTextToVideo: "Texto a video",
    tabImageToVideo: "Imagen a video",
    tabModels: "Buscar modelos",
    tabLocalModels: "Modelos locales",
    tabSettings: "Configuración",
    btnGenerateTextVideo: "Generar video desde texto",
    btnGenerateImageVideo: "Generar video desde imagen",
    btnSearchModels: "Buscar modelos",
    headingLocalModels: "Modelos locales",
    btnRefreshLocalList: "Actualizar lista local",
    btnSaveSettings: "Guardar configuración",
    statusNoTask: "No hay tareas en ejecución.",
  },
  fr: {
    languageLabel: "Langue",
    tabTextToVideo: "Texte vers vidéo",
    tabImageToVideo: "Image vers vidéo",
    tabModels: "Recherche modèles",
    tabLocalModels: "Modèles locaux",
    tabSettings: "Paramètres",
    btnGenerateTextVideo: "Générer une vidéo (texte)",
    btnGenerateImageVideo: "Générer une vidéo (image)",
    btnSearchModels: "Rechercher des modèles",
    headingLocalModels: "Modèles locaux",
    btnRefreshLocalList: "Rafraîchir la liste locale",
    btnSaveSettings: "Enregistrer",
    statusNoTask: "Aucune tâche en cours.",
  },
  de: {
    languageLabel: "Sprache",
    tabTextToVideo: "Text zu Video",
    tabImageToVideo: "Bild zu Video",
    tabModels: "Modellsuche",
    tabLocalModels: "Lokale Modelle",
    tabSettings: "Einstellungen",
    btnGenerateTextVideo: "Textvideo erstellen",
    btnGenerateImageVideo: "Bildvideo erstellen",
    btnSearchModels: "Modelle suchen",
    headingLocalModels: "Lokale Modelle",
    btnRefreshLocalList: "Lokale Liste aktualisieren",
    btnSaveSettings: "Einstellungen speichern",
    statusNoTask: "Keine laufende Aufgabe.",
  },
  it: {
    languageLabel: "Lingua",
    tabTextToVideo: "Testo in video",
    tabImageToVideo: "Immagine in video",
    tabModels: "Ricerca modelli",
    tabLocalModels: "Modelli locali",
    tabSettings: "Impostazioni",
    btnGenerateTextVideo: "Genera video da testo",
    btnGenerateImageVideo: "Genera video da immagine",
    btnSearchModels: "Cerca modelli",
    headingLocalModels: "Modelli locali",
    btnRefreshLocalList: "Aggiorna elenco locale",
    btnSaveSettings: "Salva impostazioni",
    statusNoTask: "Nessuna attività in esecuzione.",
  },
  pt: {
    languageLabel: "Idioma",
    tabTextToVideo: "Texto para vídeo",
    tabImageToVideo: "Imagem para vídeo",
    tabModels: "Buscar modelos",
    tabLocalModels: "Modelos locais",
    tabSettings: "Configurações",
    btnGenerateTextVideo: "Gerar vídeo de texto",
    btnGenerateImageVideo: "Gerar vídeo de imagem",
    btnSearchModels: "Buscar modelos",
    headingLocalModels: "Modelos locais",
    btnRefreshLocalList: "Atualizar lista local",
    btnSaveSettings: "Salvar configurações",
    statusNoTask: "Nenhuma tarefa em execução.",
  },
  ru: {
    languageLabel: "Язык",
    tabTextToVideo: "Текст в видео",
    tabImageToVideo: "Изображение в видео",
    tabModels: "Поиск моделей",
    tabLocalModels: "Локальные модели",
    tabSettings: "Настройки",
    btnGenerateTextVideo: "Сгенерировать видео из текста",
    btnGenerateImageVideo: "Сгенерировать видео из изображения",
    btnSearchModels: "Искать модели",
    headingLocalModels: "Локальные модели",
    btnRefreshLocalList: "Обновить локальный список",
    btnSaveSettings: "Сохранить настройки",
    statusNoTask: "Нет активных задач.",
  },
  ar: {
    languageLabel: "اللغة",
    tabTextToVideo: "نص إلى فيديو",
    tabImageToVideo: "صورة إلى فيديو",
    tabModels: "بحث النماذج",
    tabLocalModels: "النماذج المحلية",
    tabSettings: "الإعدادات",
    btnGenerateTextVideo: "إنشاء فيديو من النص",
    btnGenerateImageVideo: "إنشاء فيديو من الصورة",
    btnSearchModels: "بحث عن النماذج",
    headingLocalModels: "النماذج المحلية",
    btnRefreshLocalList: "تحديث القائمة المحلية",
    btnSaveSettings: "حفظ الإعدادات",
    statusNoTask: "لا توجد مهام قيد التشغيل.",
  },
};

const state = {
  currentTaskId: null,
  pollTimer: null,
  settings: null,
  localModels: [],
  localModelsBaseDir: "",
  settingsLocalModels: [],
  lastSearchResults: [],
  language: DEFAULT_LANG,
  modelCatalog: {
    "text-to-image": [],
    "image-to-image": [],
    "text-to-video": [],
    "image-to-video": [],
  },
  defaultModels: {
    "text-to-image": "",
    "image-to-image": "",
    "text-to-video": "",
    "image-to-video": "",
  },
  loraCatalog: {
    "text-to-image": [],
    "image-to-image": [],
    "text-to-video": [],
    "image-to-video": [],
  },
  vaeCatalog: {
    "text-to-image": [],
    "image-to-image": [],
  },
  localLineageFilter: "all",
  outputs: [],
  outputsBaseDir: "",
  runtimeInfo: null,
  searchPage: 1,
  searchNextCursor: null,
  searchPrevCursor: null,
  searchViewMode: "grid",
  searchDetail: null,
};

const SEARCH_BASE_MODEL_OPTIONS_BY_TASK = {
  "text-to-image": [
    "all",
    "StableDiffusion 1.x",
    "StableDiffusion 1.5",
    "StableDiffusion 2.x",
    "StableDiffusion 2.1",
    "StableDiffusion XL",
    "FLUX",
    "PixArt",
    "AuraFlow",
    "Wan",
    "Other",
  ],
  "image-to-image": [
    "all",
    "StableDiffusion 1.x",
    "StableDiffusion 1.5",
    "StableDiffusion 2.x",
    "StableDiffusion 2.1",
    "StableDiffusion XL",
    "FLUX",
    "PixArt",
    "AuraFlow",
    "Wan",
    "Other",
  ],
  "text-to-video": ["all", "TextToVideoSD", "Wan", "Other"],
  "image-to-video": ["all", "I2VGenXL", "Wan", "Other"],
};

function el(id) {
  return document.getElementById(id);
}

function bindClick(id, handler) {
  const node = el(id);
  if (!node) return;
  node.addEventListener("click", handler);
}

function readNum(id) {
  const raw = el(id).value.trim();
  if (!raw) return null;
  const parsed = Number(raw);
  return Number.isFinite(parsed) ? parsed : null;
}

function getSelectedValues(selectId) {
  const node = el(selectId);
  if (!node) return [];
  return Array.from(node.selectedOptions || [])
    .map((opt) => String(opt.value || "").trim())
    .filter((v) => v);
}

function normalizeLanguage(value) {
  const base = (value || DEFAULT_LANG).toLowerCase().split("-")[0];
  return SUPPORTED_LANGS.includes(base) ? base : DEFAULT_LANG;
}

function t(key, vars = {}) {
  const langPack = I18N[state.language] || I18N[DEFAULT_LANG];
  let template = langPack[key] ?? I18N[DEFAULT_LANG][key] ?? key;
  Object.entries(vars).forEach(([name, value]) => {
    template = template.replaceAll(`{${name}}`, String(value));
  });
  return template;
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

function formatModelSize(sizeBytes) {
  const size = Number(sizeBytes);
  if (!Number.isFinite(size) || size <= 0) return "n/a";
  const units = ["B", "KB", "MB", "GB", "TB", "PB"];
  let value = size;
  let unitIndex = 0;
  while (value >= 1024 && unitIndex < units.length - 1) {
    value /= 1024;
    unitIndex += 1;
  }
  const digits = value >= 100 || unitIndex === 0 ? 0 : 1;
  return `${value.toFixed(digits)} ${units[unitIndex]}`;
}

function formatDateTime(isoString) {
  if (!isoString) return "n/a";
  const dt = new Date(isoString);
  if (Number.isNaN(dt.getTime())) return String(isoString);
  return dt.toLocaleString();
}

function getModelOptionLabel(item) {
  const baseLabel = item.label || item.id || item.value;
  const sizeText = formatModelSize(item.size_bytes);
  if (sizeText === "n/a") return baseLabel;
  return `${baseLabel} (${t("modelSize")}: ${sizeText})`;
}

function normalizeModelId(value) {
  return String(value || "").trim().toLowerCase();
}

function getInstalledModelIdSet() {
  const installed = new Set();
  (state.localModels || []).forEach((item) => {
    installed.add(normalizeModelId(item.repo_hint));
    installed.add(normalizeModelId(item.name));
  });
  return installed;
}

function parseCivitaiId(value) {
  const raw = String(value || "").trim();
  const match = raw.match(/^civitai\/(\d+)/i) || raw.match(/^(\d+)$/);
  if (!match) return null;
  const parsed = Number(match[1]);
  return Number.isInteger(parsed) && parsed > 0 ? parsed : null;
}

function taskShortName(task) {
  if (task === "text-to-image") return "T2I";
  if (task === "image-to-image") return "I2I";
  if (task === "text-to-video") return "T2V";
  if (task === "image-to-video") return "I2V";
  return task;
}

function detectInitialLanguage() {
  const saved = localStorage.getItem(LANG_STORAGE_KEY);
  if (saved) return normalizeLanguage(saved);
  return normalizeLanguage(navigator.language || DEFAULT_LANG);
}

function applyI18n() {
  document.documentElement.lang = state.language;
  document.documentElement.dir = state.language === "ar" ? "rtl" : "ltr";
  document.title = t("appTitle");
  document.querySelectorAll("[data-i18n]").forEach((node) => {
    node.textContent = t(node.dataset.i18n);
  });
  document.querySelectorAll("[data-i18n-placeholder]").forEach((node) => {
    node.placeholder = t(node.dataset.i18nPlaceholder);
  });
  document.querySelectorAll("[data-help-key]").forEach((node) => {
    const text = t(node.dataset.helpKey);
    node.setAttribute("data-help", text);
    node.setAttribute("title", text);
  });
  refreshSearchSourceOptions();
  renderSearchBaseModelOptions();
  if (!state.currentTaskId) {
    showTaskMessage(t("statusNoTask"));
    renderTaskProgress(null);
  }
  renderModelSelect("text-to-image");
  renderModelSelect("image-to-image");
  renderModelSelect("text-to-video");
  renderModelSelect("image-to-video");
  renderLoraSelect("text-to-image");
  renderLoraSelect("image-to-image");
  renderLoraSelect("text-to-video");
  renderLoraSelect("image-to-video");
  renderVaeSelect("text-to-image");
  renderVaeSelect("image-to-image");
  renderSettingsDefaultModelSelects();
  renderLocalLineageOptions(state.localModels || []);
  renderLocalModels(state.localModels || [], state.localModelsBaseDir || "");
  renderOutputs(state.outputs || [], state.outputsBaseDir || "");
  const detailEmpty = el("modelDetailEmpty");
  if (detailEmpty && !state.searchDetail) {
    detailEmpty.textContent = t("msgModelDetailEmpty");
  }
  if (el("searchPrevBtn")) el("searchPrevBtn").textContent = t("btnPrev");
  if (el("searchNextBtn")) el("searchNextBtn").textContent = t("btnNext");
  if ((state.lastSearchResults || []).length) {
    renderSearchResults(state.lastSearchResults);
    renderSearchPagination({ page: state.searchPage, has_prev: Boolean(state.searchPrevCursor), has_next: Boolean(state.searchNextCursor) });
  }
}

function setLanguage(languageCode) {
  state.language = normalizeLanguage(languageCode);
  localStorage.setItem(LANG_STORAGE_KEY, state.language);
  el("languageSelect").value = state.language;
  applyI18n();
  loadRuntimeInfo().catch(() => {});
  loadLocalModels().catch(() => {});
}

async function api(path, options = {}) {
  const response = await fetch(path, options);
  if (!response.ok) {
    const text = await response.text();
    throw new Error(text || `HTTP ${response.status}`);
  }
  return response.json();
}

function showTaskMessage(text) {
  el("taskStatus").textContent = text;
}

function clamp01(value) {
  return Math.min(1, Math.max(0, Number(value) || 0));
}

function downloadProgressFromBytes(task) {
  if (!task || task.task_type !== "download") return null;
  const downloaded = Number(task.downloaded_bytes);
  const total = Number(task.total_bytes);
  if (!Number.isFinite(downloaded) || !Number.isFinite(total) || total <= 0) return null;
  return clamp01(downloaded / total);
}

function taskProgressValue(task) {
  const bytesRatio = downloadProgressFromBytes(task);
  if (bytesRatio != null) return bytesRatio;
  return clamp01(task?.progress);
}

function formatDuration(sec) {
  const total = Math.max(0, Math.floor(Number(sec) || 0));
  const h = Math.floor(total / 3600);
  const m = Math.floor((total % 3600) / 60);
  const s = total % 60;
  if (h > 0) return `${String(h).padStart(2, "0")}:${String(m).padStart(2, "0")}:${String(s).padStart(2, "0")}`;
  return `${String(m).padStart(2, "0")}:${String(s).padStart(2, "0")}`;
}

function taskTimeBase(task) {
  return task.started_at || task.created_at || null;
}

function estimateElapsedAndEta(task, progressValue) {
  const base = taskTimeBase(task);
  if (!base) return { elapsedSec: 0, etaSec: null };
  const started = new Date(base).getTime();
  if (!Number.isFinite(started)) return { elapsedSec: 0, etaSec: null };
  const elapsedSec = Math.max(0, (Date.now() - started) / 1000);
  const p = clamp01(progressValue);
  if (p <= 0 || p >= 1 || task.status !== "running") return { elapsedSec, etaSec: null };
  const totalSec = elapsedSec / p;
  const etaSec = Math.max(0, totalSec - elapsedSec);
  return { elapsedSec, etaSec };
}

function renderTaskProgress(task) {
  const wrap = el("taskProgressWrap");
  const bar = el("taskProgressBar");
  const label = el("taskProgressLabel");
  if (!wrap || !bar || !label) return;
  if (!task) {
    bar.style.width = "0%";
    label.textContent = "0% | ETA --:-- | ELAPSED 00:00";
    return;
  }
  const progressValue = taskProgressValue(task);
  const pct = Math.round(progressValue * 100);
  const { elapsedSec, etaSec } = estimateElapsedAndEta(task, progressValue);
  const etaText = etaSec == null ? "--:--" : formatDuration(etaSec);
  const elapsedText = formatDuration(elapsedSec);
  const downloaded = Number(task.downloaded_bytes);
  const total = Number(task.total_bytes);
  const downloadedText = Number.isFinite(downloaded) && downloaded <= 0 ? "0 B" : formatModelSize(downloaded);
  const bytesText =
    Number.isFinite(downloaded) && downloaded >= 0 && Number.isFinite(total) && total > 0
      ? ` | ${downloadedText} / ${formatModelSize(total)}`
      : "";
  bar.style.width = `${pct}%`;
  label.textContent = `${pct}%${bytesText} | ETA ${etaText} | ELAPSED ${elapsedText}`;
}

function saveLastTaskId(taskId) {
  if (taskId) {
    localStorage.setItem(TASK_STORAGE_KEY, taskId);
  } else {
    localStorage.removeItem(TASK_STORAGE_KEY);
  }
}

function setTabs() {
  document.querySelectorAll(".tab").forEach((button) => {
    button.addEventListener("click", () => {
      document.querySelectorAll(".tab").forEach((b) => b.classList.remove("active"));
      document.querySelectorAll(".panel").forEach((panel) => panel.classList.remove("active"));
      button.classList.add("active");
      el(`panel-${button.dataset.tab}`).classList.add("active");
    });
  });
}

function refreshSearchSourceOptions() {
  const taskNode = el("searchTask");
  const sourceNode = el("searchSource");
  if (!taskNode || !sourceNode) return;
  const supportsCivitai = ["text-to-image", "image-to-image"].includes(taskNode.value);
  const civitaiOption = sourceNode.querySelector('option[value="civitai"]');
  if (civitaiOption) {
    civitaiOption.disabled = !supportsCivitai;
  }
  if (!supportsCivitai && sourceNode.value === "civitai") {
    sourceNode.value = "all";
  }
}

function renderSearchBaseModelOptions() {
  const taskNode = el("searchTask");
  const select = el("searchBaseModel");
  if (!taskNode || !select) return;
  const task = taskNode.value || "text-to-image";
  const options = SEARCH_BASE_MODEL_OPTIONS_BY_TASK[task] || ["all", "Other"];
  const current = select.value || "all";
  select.innerHTML = options
    .map((value) => {
      if (value === "all") {
        return `<option value="all">${escapeHtml(t("searchBaseModelAll"))}</option>`;
      }
      return `<option value="${escapeHtml(value)}">${escapeHtml(value)}</option>`;
    })
    .join("");
  select.value = options.includes(current) ? current : "all";
}

function inferLocalLineage(item) {
  const text = `${item?.base_model || ""} ${item?.repo_hint || ""} ${item?.class_name || ""}`.toLowerCase();
  if (text.includes("stable-diffusion-xl") || text.includes("sdxl") || /\bxl\b/.test(text)) return "StableDiffusion XL";
  if (text.includes("stable-diffusion-2-1") || text.includes("v2-1") || text.includes("2.1")) return "StableDiffusion 2.1";
  if (text.includes("stable-diffusion-2") || /\bsd2\b/.test(text) || /\b2\.0\b/.test(text)) return "StableDiffusion 2.x";
  if (text.includes("stable-diffusion-1-5") || text.includes("v1-5") || text.includes("1.5")) return "StableDiffusion 1.5";
  if (text.includes("stable-diffusion-1") || /\bsd1\b/.test(text) || text.includes("1.4") || text.includes("1.0")) return "StableDiffusion 1.x";
  if (text.includes("flux")) return "FLUX";
  if (text.includes("pixart")) return "PixArt";
  if (text.includes("auraflow")) return "AuraFlow";
  if (text.includes("wan")) return "Wan";
  if (text.includes("i2vgen")) return "I2VGenXL";
  if (text.includes("texttovideosdpipeline") || text.includes("text-to-video")) return "TextToVideoSD";
  return "Other";
}

function localModelKind(item) {
  if (item?.is_lora) return t("modelKindLora");
  if (item?.is_vae) return t("modelKindVae");
  return t("modelKindBase");
}

function outputTypeLabel(kind) {
  if (kind === "image") return t("outputTypeImage");
  if (kind === "video") return t("outputTypeVideo");
  return t("outputTypeOther");
}

function renderLocalLineageOptions(items) {
  const select = el("localLineageFilter");
  if (!select) return;
  const values = Array.from(new Set((items || []).map((item) => inferLocalLineage(item)))).sort((a, b) => a.localeCompare(b));
  const options = [`<option value="all">${escapeHtml(t("localLineageAll"))}</option>`];
  values.forEach((lineage) => {
    options.push(`<option value="${escapeHtml(lineage)}">${escapeHtml(lineage)}</option>`);
  });
  select.innerHTML = options.join("");
  const desired = state.localLineageFilter || "all";
  select.value = Array.from(select.options).some((opt) => opt.value === desired) ? desired : "all";
  state.localLineageFilter = select.value;
}

function translateTaskType(taskType) {
  if (taskType === "text2image") return t("taskTypeText2Image");
  if (taskType === "image2image") return t("taskTypeImage2Image");
  if (taskType === "text2video") return t("taskTypeText2Video");
  if (taskType === "image2video") return t("taskTypeImage2Video");
  if (taskType === "download") return t("taskTypeDownload");
  return t("taskTypeUnknown");
}

function translateTaskStatus(status) {
  if (status === "queued") return t("statusQueued");
  if (status === "running") return t("statusRunning");
  if (status === "completed") return t("statusCompleted");
  if (status === "error") return t("statusError");
  return status || t("taskTypeUnknown");
}

function translateServerMessage(message) {
  const raw = (message || "").trim();
  if (!raw) return "";
  const progressSuffix = raw.match(/\(\d+\/\d+\)$/)?.[0] || "";
  if (raw.startsWith("Generating image")) {
    return `${t("serverGeneratingImage")}${progressSuffix ? ` ${progressSuffix}` : ""}`;
  }
  if (raw.startsWith("Generating frames")) {
    return `${t("serverGeneratingFrames")}${progressSuffix ? ` ${progressSuffix}` : ""}`;
  }
  const map = {
    Queued: t("serverQueued"),
    "Generation queued": t("serverGenerationQueued"),
    "Download queued": t("serverDownloadQueued"),
    "Loading model": t("serverLoadingModel"),
    "Applying LoRA": t("serverLoadingLora"),
    "Preparing image": t("serverPreparingImage"),
    "Generating image": t("serverGeneratingImage"),
    "Generating frames": t("serverGeneratingFrames"),
    "Decoding latents": t("serverDecodingLatents"),
    "Decoding latents (CPU fallback)": t("serverDecodingLatents"),
    "Postprocessing image": t("serverPostprocessingImage"),
    "Encoding mp4": t("serverEncoding"),
    "Saving png": t("serverSavingPng"),
    Done: t("serverDone"),
    "Generation failed": t("serverGenerationFailed"),
    "Download complete": t("serverDownloadComplete"),
    "Download failed": t("serverDownloadFailed"),
  };
  return map[raw] || raw;
}

async function loadRuntimeInfo() {
  try {
    const info = await api("/api/system/info");
    state.runtimeInfo = info;
    const flags = [
      `${t("runtimeDevice")}=${info.device}`,
      `${t("runtimeCuda")}=${info.cuda_available}`,
      `${t("runtimeRocm")}=${info.rocm_available}`,
      `${t("runtimeNpu")}=${info.npu_available}`,
      `${t("runtimeDiffusers")}=${info.diffusers_ready}`,
    ];
    if (info.torch_version) flags.push(`${t("runtimeTorch")}=${info.torch_version}`);
    if (info.npu_reason) flags.push(`${t("runtimeNpuReason")}=${info.npu_reason}`);
    if (info.import_error) flags.push(`${t("runtimeError")}=${info.import_error}`);
    el("runtimeInfo").textContent = flags.join(" | ");
    applyNpuAvailability(info);
  } catch (error) {
    state.runtimeInfo = null;
    el("runtimeInfo").textContent = t("runtimeLoadFailed", { error: error.message });
    applyNpuAvailability(null);
  }
}

function applyNpuAvailability(info) {
  const npuAvailable = Boolean(info?.npu_available);
  const npuRunnable = Boolean(info?.t2v_npu_runner_configured);
  const t2vBackend = el("t2vBackendSelect");
  if (t2vBackend) {
    const npuOption = Array.from(t2vBackend.options).find((opt) => opt.value === "npu");
    if (npuOption) npuOption.disabled = !npuRunnable;
    if (!npuRunnable && t2vBackend.value === "npu") {
      t2vBackend.value = "auto";
    }
  }
  const cfgBackend = el("cfgT2VBackend");
  if (cfgBackend) {
    const npuOption = Array.from(cfgBackend.options).find((opt) => opt.value === "npu");
    if (npuOption) npuOption.disabled = !npuRunnable;
    if (!npuRunnable && cfgBackend.value === "npu") {
      cfgBackend.value = "auto";
    }
  }
}

function applySettings(settings) {
  state.settings = settings;
  state.defaultModels["text-to-image"] = settings.defaults.text2image_model || "";
  state.defaultModels["image-to-image"] = settings.defaults.image2image_model || "";
  state.defaultModels["text-to-video"] = settings.defaults.text2video_model || "";
  state.defaultModels["image-to-video"] = settings.defaults.image2video_model || "";
  const serverSettings = settings.server || {};
  const supportedBackends = ["auto", "cuda", "npu"];
  const defaultT2vBackend = supportedBackends.includes(String(serverSettings.t2v_backend || "").toLowerCase())
    ? String(serverSettings.t2v_backend).toLowerCase()
    : "auto";
  el("cfgModelsDir").value = settings.paths.models_dir;
  el("cfgListenPort").value = serverSettings?.listen_port ?? 8000;
  if (el("cfgRocmAotriton")) {
    el("cfgRocmAotriton").checked = serverSettings?.rocm_aotriton_experimental !== false;
  }
  if (el("cfgT2VBackend")) {
    el("cfgT2VBackend").value = defaultT2vBackend;
  }
  if (el("cfgT2VNpuRunner")) {
    el("cfgT2VNpuRunner").value = serverSettings?.t2v_npu_runner || "";
  }
  if (el("cfgT2VNpuModelDir")) {
    el("cfgT2VNpuModelDir").value = serverSettings?.t2v_npu_model_dir || "";
  }
  if (el("t2vBackendSelect")) {
    el("t2vBackendSelect").value = defaultT2vBackend;
  }
  el("cfgOutputsDir").value = settings.paths.outputs_dir;
  el("cfgTmpDir").value = settings.paths.tmp_dir;
  el("cfgLogLevel").value = (settings.logging?.level || "INFO").toUpperCase();
  el("cfgToken").value = settings.huggingface.token || "";
  renderSettingsDefaultModelSelects(
    settings.defaults.text2video_model || "",
    settings.defaults.image2video_model || "",
    settings.defaults.text2image_model || "",
    settings.defaults.image2image_model || "",
  );
  el("cfgSteps").value = settings.defaults.num_inference_steps;
  el("cfgFrames").value = settings.defaults.num_frames;
  el("cfgGuidance").value = settings.defaults.guidance_scale;
  el("cfgFps").value = settings.defaults.fps;
  el("cfgWidth").value = settings.defaults.width;
  el("cfgHeight").value = settings.defaults.height;
  el("t2iSteps").value = settings.defaults.num_inference_steps;
  el("t2iGuidance").value = settings.defaults.guidance_scale;
  el("t2iWidth").value = settings.defaults.width;
  el("t2iHeight").value = settings.defaults.height;
  el("i2iSteps").value = settings.defaults.num_inference_steps;
  el("i2iGuidance").value = settings.defaults.guidance_scale;
  el("i2iWidth").value = settings.defaults.width;
  el("i2iHeight").value = settings.defaults.height;
  el("t2vSteps").value = settings.defaults.num_inference_steps;
  el("t2vFrames").value = settings.defaults.num_frames;
  el("t2vGuidance").value = settings.defaults.guidance_scale;
  el("t2vFps").value = settings.defaults.fps;
  el("i2vSteps").value = settings.defaults.num_inference_steps;
  el("i2vFrames").value = settings.defaults.num_frames;
  el("i2vGuidance").value = settings.defaults.guidance_scale;
  el("i2vFps").value = settings.defaults.fps;
  el("i2vWidth").value = settings.defaults.width;
  el("i2vHeight").value = settings.defaults.height;
  if (!el("downloadTargetDir").value.trim()) {
    el("downloadTargetDir").value = settings.paths.models_dir;
  }
  if (!el("localModelsDir").value.trim()) {
    el("localModelsDir").value = settings.paths.models_dir;
  }
  renderModelSelect("text-to-image");
  renderModelSelect("image-to-image");
  renderModelSelect("text-to-video");
  renderModelSelect("image-to-video");
  applyNpuAvailability(state.runtimeInfo);
}

function renderDefaultModelSettingSelect(selectId, selectedValue) {
  const select = el(selectId);
  if (!select) return;
  const taskBySelect = {
    cfgTextModel: "text-to-video",
    cfgImageModel: "image-to-video",
    cfgTextImageModel: "text-to-image",
    cfgImageImageModel: "image-to-image",
  };
  const task = taskBySelect[selectId];
  const localIds = Array.from(new Set((state.modelCatalog[task] || []).map((item) => item.id || "")))
    .filter((id) => id)
    .sort((a, b) => String(a).localeCompare(String(b)));
  const normalizedSelected = String(selectedValue || "").trim();
  const options = [`<option value="">${escapeHtml(t("msgSelectLocalModel"))}</option>`];
  localIds.forEach((modelId) => {
    options.push(`<option value="${escapeHtml(modelId)}">${escapeHtml(modelId)}</option>`);
  });
  if (normalizedSelected && !localIds.includes(normalizedSelected)) {
    options.push(
      `<option value="${escapeHtml(normalizedSelected)}">${escapeHtml(t("msgDefaultModelNotLocal", { model: normalizedSelected }))}</option>`,
    );
  }
  select.innerHTML = options.join("");
  if (normalizedSelected && Array.from(select.options).some((opt) => opt.value === normalizedSelected)) {
    select.value = normalizedSelected;
  } else {
    select.value = "";
  }
}

function renderSettingsDefaultModelSelects(textModelValue = null, imageModelValue = null, textImageModelValue = null, imageImageModelValue = null) {
  const textValue = textModelValue !== null ? textModelValue : el("cfgTextModel")?.value || state.settings?.defaults?.text2video_model || "";
  const imageValue =
    imageModelValue !== null ? imageModelValue : el("cfgImageModel")?.value || state.settings?.defaults?.image2video_model || "";
  const textImageValue =
    textImageModelValue !== null
      ? textImageModelValue
      : el("cfgTextImageModel")?.value || state.settings?.defaults?.text2image_model || "";
  const imageImageValue =
    imageImageModelValue !== null
      ? imageImageModelValue
      : el("cfgImageImageModel")?.value || state.settings?.defaults?.image2image_model || "";
  renderDefaultModelSettingSelect("cfgTextModel", textValue);
  renderDefaultModelSettingSelect("cfgImageModel", imageValue);
  renderDefaultModelSettingSelect("cfgTextImageModel", textImageValue);
  renderDefaultModelSettingSelect("cfgImageImageModel", imageImageValue);
}

async function loadSettingsLocalModels() {
  await Promise.all([
    loadModelCatalog("text-to-image", true),
    loadModelCatalog("image-to-image", true),
    loadModelCatalog("text-to-video", true),
    loadModelCatalog("image-to-video", true),
  ]);
  renderSettingsDefaultModelSelects();
}

async function loadSettings() {
  const settings = await api("/api/settings");
  applySettings(settings);
  await loadSettingsLocalModels();
}

function getModelDom(task) {
  if (task === "text-to-image") {
    return { selectId: "t2iModelSelect", previewId: "t2iModelPreview" };
  }
  if (task === "image-to-image") {
    return { selectId: "i2iModelSelect", previewId: "i2iModelPreview" };
  }
  if (task === "text-to-video") {
    return { selectId: "t2vModelSelect", previewId: "t2vModelPreview" };
  }
  return { selectId: "i2vModelSelect", previewId: "i2vModelPreview" };
}

function getLoraDom(task) {
  if (task === "text-to-image") return { selectId: "t2iLoraSelect" };
  if (task === "image-to-image") return { selectId: "i2iLoraSelect" };
  if (task === "text-to-video") return { selectId: "t2vLoraSelect" };
  return { selectId: "i2vLoraSelect" };
}

function getVaeDom(task) {
  if (task === "text-to-image") return { selectId: "t2iVaeSelect" };
  return { selectId: "i2iVaeSelect" };
}

function getSelectedOrDefaultModelValue(task) {
  const modelDom = getModelDom(task);
  const selected = (el(modelDom.selectId)?.value || "").trim();
  if (selected) return selected;
  return (state.defaultModels[task] || "").trim();
}

function renderLoraSelect(task, preferredValue = null) {
  const dom = getLoraDom(task);
  const select = el(dom.selectId);
  if (!select) return;
  const currentValues = preferredValue === null ? getSelectedValues(dom.selectId) : preferredValue;
  const items = state.loraCatalog[task] || [];
  let html = `<option value="">${escapeHtml(t("msgNoLoraOption"))}</option>`;
  html += items.map((item) => `<option value="${escapeHtml(item.value)}">${escapeHtml(getModelOptionLabel(item))}</option>`).join("");
  select.innerHTML = html;
  const currentSet = new Set(currentValues || []);
  Array.from(select.options).forEach((option, idx) => {
    option.selected = currentSet.has(option.value);
    if (!currentSet.size && idx === 0) {
      option.selected = true;
    }
  });
}

async function loadLoraCatalog(task, keepSelection = true) {
  const dom = getLoraDom(task);
  const select = el(dom.selectId);
  const prev = keepSelection && select ? getSelectedValues(dom.selectId) : [];
  const params = new URLSearchParams({
    task,
    limit: "200",
  });
  const modelRef = getSelectedOrDefaultModelValue(task);
  if (modelRef) params.set("model_ref", modelRef);
  const data = await api(`/api/models/loras/catalog?${params.toString()}`);
  state.loraCatalog[task] = data.items || [];
  renderLoraSelect(task, prev);
}

function renderVaeSelect(task, preferredValue = null) {
  const dom = getVaeDom(task);
  const select = el(dom.selectId);
  if (!select) return;
  const currentValue = preferredValue === null ? select.value : preferredValue;
  const items = state.vaeCatalog[task] || [];
  let html = `<option value="">${escapeHtml(t("msgNoVaeOption"))}</option>`;
  html += items.map((item) => `<option value="${escapeHtml(item.value)}">${escapeHtml(getModelOptionLabel(item))}</option>`).join("");
  select.innerHTML = html;
  if (currentValue && items.some((item) => item.value === currentValue)) {
    select.value = currentValue;
  } else {
    select.value = "";
  }
}

async function loadVaeCatalog(task, keepSelection = true) {
  const dom = getVaeDom(task);
  const select = el(dom.selectId);
  const prev = keepSelection && select ? select.value : "";
  const data = await api("/api/models/vaes/catalog?limit=200");
  state.vaeCatalog[task] = data.items || [];
  renderVaeSelect(task, prev);
}

function renderModelPreview(task) {
  const dom = getModelDom(task);
  const select = el(dom.selectId);
  const preview = el(dom.previewId);
  const selectedValue = select.value;
  const catalog = state.modelCatalog[task] || [];
  const defaultId = state.defaultModels[task] || "";

  let chosen = null;
  if (selectedValue) {
    chosen = catalog.find((item) => item.value === selectedValue) || null;
  } else if (defaultId) {
    chosen = catalog.find((item) => item.id === defaultId || item.value === defaultId) || null;
  }

  if (!chosen) {
    preview.innerHTML = `<p>${t("msgModelSelectHint")}</p>`;
    return;
  }

  const name = escapeHtml(chosen.id || chosen.label || selectedValue || defaultId);
  const modelUrl = chosen.model_url ? escapeHtml(chosen.model_url) : "";
  const infoHtml = modelUrl
    ? `<strong><a href="${modelUrl}" target="_blank" rel="noopener noreferrer">${name}</a></strong>`
    : `<strong>${name}</strong>`;

  const imageHtml = chosen.preview_url
    ? `<img class="model-picked-thumb" src="${escapeHtml(chosen.preview_url)}" alt="${escapeHtml(t("msgModelPreviewAlt"))}" loading="lazy" onerror="this.style.display='none'" />`
    : `<div class="model-picked-empty">${escapeHtml(t("msgModelNoPreview"))}</div>`;
  const sizeText = formatModelSize(chosen.size_bytes);
  const metaText =
    sizeText === "n/a"
      ? escapeHtml(chosen.source || "model")
      : `${escapeHtml(chosen.source || "model")} | ${escapeHtml(t("modelSize"))}: ${escapeHtml(sizeText)}`;

  preview.innerHTML = `
    <div class="model-picked-card">
      ${imageHtml}
      <div class="model-picked-meta">${infoHtml}<span>${metaText}</span></div>
    </div>
  `;
}

function renderModelSelect(task, preferredValue = null) {
  const dom = getModelDom(task);
  const select = el(dom.selectId);
  const currentValue = preferredValue === null ? select.value : preferredValue;
  const items = state.modelCatalog[task] || [];
  const defaultId = state.defaultModels[task] || "";
  const defaultLabel = defaultId ? t("msgDefaultModelOption", { model: defaultId }) : t("msgDefaultModelNoMeta");

  let html = `<option value="">${escapeHtml(defaultLabel)}</option>`;
  html += items
    .map((item) => `<option value="${escapeHtml(item.value)}">${escapeHtml(getModelOptionLabel(item))}</option>`)
    .join("");
  select.innerHTML = html;

  if (currentValue && items.some((item) => item.value === currentValue)) {
    select.value = currentValue;
  } else {
    select.value = "";
  }
  renderModelPreview(task);
}

async function loadModelCatalog(task, keepSelection = true) {
  const dom = getModelDom(task);
  const prev = keepSelection ? el(dom.selectId).value : "";
  const params = new URLSearchParams({
    task,
    limit: "40",
  });
  const data = await api(`/api/models/catalog?${params.toString()}`);
  state.modelCatalog[task] = data.items || [];
  state.defaultModels[task] = data.default_model || state.defaultModels[task] || "";
  renderModelSelect(task, prev);
  await loadLoraCatalog(task, keepSelection);
  if (task === "text-to-image" || task === "image-to-image") {
    await loadVaeCatalog(task, keepSelection);
  }
}

async function saveSettings(event) {
  event.preventDefault();
  const prevPort = Number(state.settings?.server?.listen_port || 0);
  const prevRocmAotriton = state.settings?.server?.rocm_aotriton_experimental !== false;
  const payload = {
    server: {
      listen_port: Number(el("cfgListenPort").value),
      rocm_aotriton_experimental: Boolean(el("cfgRocmAotriton")?.checked),
      t2v_backend: (el("cfgT2VBackend")?.value || "auto").trim(),
      t2v_npu_runner: (el("cfgT2VNpuRunner")?.value || "").trim(),
      t2v_npu_model_dir: (el("cfgT2VNpuModelDir")?.value || "").trim(),
    },
    paths: {
      models_dir: el("cfgModelsDir").value.trim(),
      outputs_dir: el("cfgOutputsDir").value.trim(),
      tmp_dir: el("cfgTmpDir").value.trim(),
    },
    logging: {
      level: el("cfgLogLevel").value.trim() || "INFO",
    },
    huggingface: {
      token: el("cfgToken").value.trim(),
    },
    defaults: {
      text2image_model: el("cfgTextImageModel").value.trim(),
      image2image_model: el("cfgImageImageModel").value.trim(),
      text2video_model: el("cfgTextModel").value.trim(),
      image2video_model: el("cfgImageModel").value.trim(),
      num_inference_steps: Number(el("cfgSteps").value),
      num_frames: Number(el("cfgFrames").value),
      guidance_scale: Number(el("cfgGuidance").value),
      fps: Number(el("cfgFps").value),
      width: Number(el("cfgWidth").value),
      height: Number(el("cfgHeight").value),
    },
  };
  const updated = await api("/api/settings", {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  applySettings(updated);
  await Promise.all([loadSettingsLocalModels(), loadLocalModels()]);
  await Promise.all([
    loadModelCatalog("text-to-image", true),
    loadModelCatalog("image-to-image", true),
    loadModelCatalog("text-to-video", true),
    loadModelCatalog("image-to-video", true),
  ]);
  const newPort = Number(updated.server?.listen_port || 0);
  const newRocmAotriton = updated.server?.rocm_aotriton_experimental !== false;
  if ((prevPort && newPort && prevPort !== newPort) || prevRocmAotriton !== newRocmAotriton) {
    if (prevPort && newPort && prevPort !== newPort) {
      showTaskMessage(t("msgPortChangeSaved"));
    } else {
      showTaskMessage(t("msgServerSettingRestartRequired"));
    }
  } else {
    showTaskMessage(t("msgSettingsSaved"));
  }
}

async function clearHfCache() {
  if (!window.confirm(t("msgConfirmClearHfCache"))) return;
  const result = await api("/api/cache/hf/clear", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ dry_run: false }),
  });
  showTaskMessage(
    t("msgHfCacheCleared", {
      removed: (result.removed_paths || []).length,
      skipped: (result.skipped || []).length,
      failed: (result.failed || []).length,
    }),
  );
}

async function deleteLocalModel(item, baseDir = "") {
  const name = item?.repo_hint || item?.name || "";
  if (!window.confirm(t("msgConfirmDeleteModel", { model: name }))) return;
  await api("/api/models/local/delete", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      model_name: item.name,
      base_dir: baseDir || null,
    }),
  });
  await Promise.all([loadLocalModels(), loadSettingsLocalModels()]);
  await Promise.all([
    loadModelCatalog("text-to-image", true),
    loadModelCatalog("image-to-image", true),
    loadModelCatalog("text-to-video", true),
    loadModelCatalog("image-to-video", true),
  ]);
  showTaskMessage(t("msgModelDeleted", { model: name }));
}

function renderLocalModels(items, baseDir = "") {
  const container = el("localModels");
  const activeLineage = state.localLineageFilter || "all";
  const filteredItems =
    activeLineage === "all" ? [...(items || [])] : (items || []).filter((item) => inferLocalLineage(item) === activeLineage);
  if (!filteredItems.length) {
    container.innerHTML = `<p>${t("msgNoLocalModels", { path: baseDir || t("msgUnknownPath") })}</p>`;
    return;
  }
  const taskShortLabel = {
    "text-to-image": "T2I",
    "image-to-image": "I2I",
    "text-to-video": "T2V",
    "image-to-video": "I2V",
  };
  container.innerHTML = filteredItems
    .map(
      (item, index) => `
      <div class="row model-row">
        <div class="model-main">
          ${
            item.preview_url
              ? `<img class="model-preview" src="${escapeHtml(item.preview_url)}" alt="${escapeHtml(t("msgModelPreviewAlt"))}" loading="lazy" onerror="this.style.display='none'" />`
              : ""
          }
          <div>
            <strong>${escapeHtml(item.repo_hint)}</strong>
            <span>${escapeHtml(t("modelKind"))}=${escapeHtml(localModelKind(item))} | ${escapeHtml(t("modelTag"))}=${escapeHtml(item.class_name || "n/a")} | ${escapeHtml(t("modelSource"))}=local</span>
            <span>lineage=${escapeHtml(inferLocalLineage(item))} | ${escapeHtml(t("modelBase"))}=${escapeHtml(item.base_model || "n/a")}</span>
          </div>
        </div>
        <div class="local-actions">
          ${(item.compatible_tasks || [])
            .map(
              (task) =>
                `<button type="button" class="local-apply-btn" data-index="${index}" data-task="${escapeHtml(task)}">${escapeHtml(
                  t("msgSetTaskModel", { task: taskShortLabel[task] || task }),
                )}</button>`,
            )
            .join("")}
          ${item.can_delete ? `<button type="button" class="local-delete-btn" data-index="${index}">${t("btnDeleteModel")}</button>` : ""}
        </div>
      </div>`,
    )
    .join("");
  container.querySelectorAll(".local-apply-btn").forEach((button) => {
    button.addEventListener("click", async () => {
      const index = Number(button.dataset.index || "-1");
      const task = button.dataset.task || "";
      if (!Number.isInteger(index) || index < 0 || index >= filteredItems.length) return;
      const item = filteredItems[index];
      const modelDom = getModelDom(task);
      const select = el(modelDom.selectId);
      if (!select) return;
      const catalog = state.modelCatalog[task] || [];
      if (!catalog.some((entry) => entry.value === item.path)) {
        catalog.push({
          source: "local",
          label: `[local] ${item.repo_hint}`,
          value: item.path,
          id: item.repo_hint,
          size_bytes: null,
          preview_url: item.preview_url || null,
          model_url: item.repo_hint ? `https://huggingface.co/${encodeURIComponent(item.repo_hint).replaceAll("%2F", "/")}` : null,
        });
      }
      state.modelCatalog[task] = catalog;
      renderModelSelect(task, item.path);
      try {
        await loadLoraCatalog(task, false);
        if (task === "text-to-image" || task === "image-to-image") {
          await loadVaeCatalog(task, false);
        }
      } catch (error) {
        showTaskMessage(t("msgSearchFailed", { error: error.message }));
      }
      showTaskMessage(t("msgLocalModelApplied", { task, model: item.repo_hint }));
    });
  });
  container.querySelectorAll(".local-delete-btn").forEach((button) => {
    button.addEventListener("click", async () => {
      const index = Number(button.dataset.index || "-1");
      if (!Number.isInteger(index) || index < 0 || index >= filteredItems.length) return;
      try {
        await deleteLocalModel(filteredItems[index], baseDir);
      } catch (error) {
        showTaskMessage(t("msgModelDeleteFailed", { error: error.message }));
      }
    });
  });
}

async function loadLocalModels() {
  const dir = el("localModelsDir")?.value?.trim() || "";
  const params = new URLSearchParams();
  if (dir) params.set("dir", dir);
  const url = params.toString() ? `/api/models/local?${params.toString()}` : "/api/models/local";
  const data = await api(url);
  state.localModels = data.items || [];
  state.localModelsBaseDir = data.base_dir || dir || "";
  renderLocalLineageOptions(state.localModels);
  renderLocalModels(state.localModels, state.localModelsBaseDir);
  if ((state.lastSearchResults || []).length) {
    renderSearchResults(state.lastSearchResults);
  }
}

function renderOutputs(items, baseDir = "") {
  const pathNode = el("outputsPath");
  if (pathNode) {
    pathNode.textContent = baseDir || "";
  }
  const container = el("outputsList");
  if (!container) return;
  if (!items.length) {
    container.innerHTML = `<p>${t("msgNoOutputs", { path: baseDir || t("msgUnknownPath") })}</p>`;
    return;
  }
  container.innerHTML = items
    .map((item, index) => {
      const viewUrl = item.view_url || "";
      let previewHtml = "";
      if (item.kind === "image" && viewUrl) {
        previewHtml = `<img class="model-preview" src="${escapeHtml(viewUrl)}" alt="${escapeHtml(item.name || "output")}" loading="lazy" onerror="this.style.display='none'" />`;
      } else if (item.kind === "video" && viewUrl) {
        previewHtml = `<video class="model-preview" src="${escapeHtml(viewUrl)}" preload="metadata" muted playsinline></video>`;
      }
      const openLink = viewUrl ? `<a href="${escapeHtml(viewUrl)}" target="_blank" rel="noopener noreferrer">${t("msgOpen")}</a>` : "";
      return `
      <div class="row model-row">
        <div class="model-main">
          ${previewHtml}
          <div>
            <strong>${escapeHtml(item.name || "")}</strong>
            <span>${escapeHtml(t("modelSize"))}=${escapeHtml(formatModelSize(item.size_bytes))} | ${escapeHtml(t("outputUpdated"))}=${escapeHtml(formatDateTime(item.updated_at))} | ${escapeHtml(t("modelTag"))}=${escapeHtml(outputTypeLabel(item.kind))}</span>
          </div>
        </div>
        <div class="local-actions">
          ${openLink}
          <button type="button" class="output-delete-btn" data-index="${index}">${t("btnDeleteModel")}</button>
        </div>
      </div>`;
    })
    .join("");
  container.querySelectorAll(".output-delete-btn").forEach((button) => {
    button.addEventListener("click", async () => {
      const index = Number(button.dataset.index || "-1");
      if (!Number.isInteger(index) || index < 0 || index >= items.length) return;
      try {
        await deleteOutput(items[index]);
      } catch (error) {
        showTaskMessage(t("msgOutputDeleteFailed", { error: error.message }));
      }
    });
  });
}

async function loadOutputs() {
  const data = await api("/api/outputs?limit=500");
  state.outputs = data.items || [];
  state.outputsBaseDir = data.base_dir || "";
  renderOutputs(state.outputs, state.outputsBaseDir);
}

async function deleteOutput(item) {
  const name = String(item?.name || "").trim();
  if (!name) return;
  if (!window.confirm(t("msgConfirmDeleteOutput", { name }))) return;
  await api("/api/outputs/delete", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ file_name: name }),
  });
  await loadOutputs();
  showTaskMessage(t("msgOutputDeleted", { name }));
}

function searchItemInstalled(item) {
  if (item?.installed === true) return true;
  const installed = getInstalledModelIdSet();
  return installed.has(normalizeModelId(item?.id));
}

function renderSearchPagination(pageInfo = null) {
  const prevBtn = el("searchPrevBtn");
  const nextBtn = el("searchNextBtn");
  const label = el("searchPageInfo");
  const page = Number(pageInfo?.page || state.searchPage || 1);
  if (label) {
    label.textContent = t("msgSearchPage", { page });
  }
  if (prevBtn) prevBtn.disabled = !(pageInfo?.has_prev || state.searchPrevCursor);
  if (nextBtn) nextBtn.disabled = !(pageInfo?.has_next || state.searchNextCursor);
}

function selectedDetailDownloadOptions(item) {
  const detail = state.searchDetail;
  if (!detail || normalizeModelId(detail.item?.id) !== normalizeModelId(item?.id)) {
    return {};
  }
  if (detail.item.source === "huggingface") {
    const revisionInput = el("detailHfRevisionInput");
    const revisionSelect = el("detailHfRevision");
    const revision = (revisionInput?.value || revisionSelect?.value || "main").trim() || "main";
    return {
      source: "huggingface",
      hf_revision: revision,
    };
  }
  if (detail.item.source === "civitai") {
    const modelId = parseCivitaiId(detail.item.id);
    const versionId = Number(el("detailVersionSelect")?.value || "");
    const fileId = Number(el("detailFileSelect")?.value || "");
    return {
      source: "civitai",
      civitai_model_id: modelId || null,
      civitai_version_id: Number.isInteger(versionId) && versionId > 0 ? versionId : null,
      civitai_file_id: Number.isInteger(fileId) && fileId > 0 ? fileId : null,
    };
  }
  return {};
}

function buildDetailFiles(version) {
  const files = Array.isArray(version?.files) ? version.files : [];
  if (!files.length) return `<option value="">-</option>`;
  return files
    .map((file) => `<option value="${escapeHtml(file.id)}">${escapeHtml(file.name)} (${escapeHtml(formatModelSize(file.size))})</option>`)
    .join("");
}

function renderModelDetail(item, detail) {
  const empty = el("modelDetailEmpty");
  const content = el("modelDetailContent");
  if (!content || !empty) return;
  empty.style.display = "none";
  const previews = Array.isArray(detail.previews) ? detail.previews.filter(Boolean) : [];
  const tags = Array.isArray(detail.tags) ? detail.tags : [];
  const versions = Array.isArray(detail.versions) ? detail.versions : [];
  const defaultVersionId = detail.default_version_id != null ? String(detail.default_version_id) : "";
  let selectedVersion = versions.find((v) => String(v.id) === defaultVersionId) || versions[0] || null;
  const hfRevision = selectedVersion?.name || "main";
  const description = String(detail.description || "").trim();
  const sourceUrl = item.model_url || detail.model_url || "#";
  const modelId = String(item.id || detail.id || "");
  content.classList.add("active");
  content.innerHTML = `
    <div class="model-detail-head">
      <h4 class="model-detail-title">${escapeHtml(detail.title || item.title || item.id || "")}</h4>
      <div class="model-detail-id">${escapeHtml(modelId)}</div>
      <div class="model-detail-meta">${escapeHtml(t("modelSource"))}: ${escapeHtml(item.source || detail.source || "")}</div>
    </div>
    <div class="model-detail-actions">
      <a href="${escapeHtml(sourceUrl)}" target="_blank" rel="noopener noreferrer">${escapeHtml(t("msgOpen"))}</a>
      <button id="detailDownloadBtn" type="button">${escapeHtml(t("btnDownload"))}</button>
      <button id="detailApplyBtn" type="button">${escapeHtml(t("msgApply"))}</button>
    </div>
    <div>
      <strong>${escapeHtml(t("msgDetailDescription"))}</strong>
      <div class="model-detail-text">${escapeHtml(description || "-")}</div>
    </div>
    <div>
      <strong>${escapeHtml(t("msgDetailTags"))}</strong>
      <div class="model-detail-tags">${tags.slice(0, 30).map((tag) => `<span class="model-detail-tag">${escapeHtml(tag)}</span>`).join("") || "-"}</div>
    </div>
    <div>
      <strong>${escapeHtml(t("msgModelPreviewAlt"))}</strong>
      <div class="model-detail-gallery">${previews
        .slice(0, 12)
        .map((url) => `<img src="${escapeHtml(url)}" alt="${escapeHtml(t("msgModelPreviewAlt"))}" loading="lazy" onerror="this.style.display='none'" />`)
        .join("")}</div>
    </div>
    <div class="model-detail-grid">
      ${
        item.source === "huggingface"
          ? `
      <label>
        <span>${escapeHtml(t("msgDetailRevision"))}</span>
        <select id="detailHfRevision">${versions
          .map((version) => `<option value="${escapeHtml(version.name)}">${escapeHtml(version.name)}</option>`)
          .join("")}</select>
      </label>
      <label>
        <span>${escapeHtml(t("msgDetailRevision"))} (manual)</span>
        <input id="detailHfRevisionInput" value="${escapeHtml(hfRevision)}" />
      </label>
      `
          : `
      <label>
        <span>${escapeHtml(t("msgDetailVersions"))}</span>
        <select id="detailVersionSelect">${versions
          .map((version) => `<option value="${escapeHtml(version.id)}">${escapeHtml(version.name || version.id)}</option>`)
          .join("")}</select>
      </label>
      <label>
        <span>${escapeHtml(t("msgDetailFiles"))}</span>
        <select id="detailFileSelect">${buildDetailFiles(selectedVersion)}</select>
      </label>
      `
      }
    </div>
  `;

  if (item.source === "huggingface") {
    const revisionSelect = el("detailHfRevision");
    const revisionInput = el("detailHfRevisionInput");
    if (revisionSelect) {
      revisionSelect.value = hfRevision;
      revisionSelect.addEventListener("change", () => {
        if (revisionInput) revisionInput.value = revisionSelect.value;
      });
    }
  } else {
    const versionSelect = el("detailVersionSelect");
    const fileSelect = el("detailFileSelect");
    if (versionSelect) {
      versionSelect.value = selectedVersion ? String(selectedVersion.id) : "";
      versionSelect.addEventListener("change", () => {
        const nextVersion = versions.find((version) => String(version.id) === versionSelect.value) || null;
        selectedVersion = nextVersion;
        if (fileSelect) {
          fileSelect.innerHTML = buildDetailFiles(nextVersion);
        }
      });
    }
  }
  bindClick("detailDownloadBtn", async () => {
    await startModelDownload(item.id, selectedDetailDownloadOptions(item));
  });
  bindClick("detailApplyBtn", async () => {
    await applySearchResultModel(item);
  });
}

async function openModelDetail(item) {
  const empty = el("modelDetailEmpty");
  const content = el("modelDetailContent");
  if (empty) {
    empty.style.display = "block";
    empty.textContent = t("msgModelDetailLoading");
  }
  if (content) {
    content.classList.remove("active");
    content.innerHTML = "";
  }
  const source = item?.source === "civitai" ? "civitai" : "huggingface";
  try {
    const params = new URLSearchParams({
      source,
      id: String(item.id || ""),
    });
    const detail = await api(`/api/models/detail?${params.toString()}`);
    state.searchDetail = { item, detail };
    renderModelDetail(item, detail);
  } catch (error) {
    state.searchDetail = null;
    if (empty) {
      empty.style.display = "block";
      empty.textContent = t("msgModelDetailLoadFailed", { error: error.message });
    }
  }
}

async function applySearchResultModel(item) {
  const task = el("searchTask").value;
  const modelDom = getModelDom(task);
  const select = el(modelDom.selectId);
  if (!select) return;
  const catalog = state.modelCatalog[task] || [];
  const value = String(item.id || "").trim();
  if (!value) return;
  if (!catalog.some((entry) => entry.value === value)) {
    catalog.push({
      source: item.source || "remote",
      label: `[${item.source || "remote"}] ${item.id}`,
      value,
      id: item.id,
      size_bytes: item.size_bytes || null,
      preview_url: item.preview_url || null,
      model_url: item.model_url || null,
    });
  }
  state.modelCatalog[task] = catalog;
  renderModelSelect(task, value);
  try {
    await loadLoraCatalog(task, false);
    if (task === "text-to-image" || task === "image-to-image") {
      await loadVaeCatalog(task, false);
    }
  } catch (error) {
    showTaskMessage(t("msgSearchFailed", { error: error.message }));
  }
  showTaskMessage(t("msgSearchModelApplied", { task: taskShortName(task), model: item.id }));
}

function renderSearchResults(items) {
  const cards = el("searchCards");
  const viewMode = (el("searchViewMode")?.value || "grid").trim();
  state.searchViewMode = viewMode === "list" ? "list" : "grid";
  if (!cards) return;
  cards.classList.toggle("list-mode", state.searchViewMode === "list");
  if (!items.length) {
    cards.innerHTML = `<p>${escapeHtml(t("msgNoModelsFound"))}</p>`;
    renderSearchPagination({ page: state.searchPage, has_prev: false, has_next: false });
    return;
  }

  cards.innerHTML = items
    .map((item, index) => {
      const installed = searchItemInstalled(item);
      const supportsDownload = item.download_supported !== false;
      const preview = item.preview_url
        ? `<img class="model-card-cover" src="${escapeHtml(item.preview_url)}" alt="${escapeHtml(t("msgModelPreviewAlt"))}" loading="lazy" onerror="this.style.display='none'" />`
        : `<img class="model-card-cover" src="" alt="${escapeHtml(t("msgModelPreviewAlt"))}" style="display:none" />`;
      const statusBadge = installed
        ? `<span class="model-status-badge downloaded">${escapeHtml(t("msgModelInstalled"))}</span>`
        : `<span class="model-status-badge">${escapeHtml(t("msgModelNotInstalled"))}</span>`;
      return `
        <article class="model-card ${state.searchViewMode === "list" ? "list-mode" : ""}" data-index="${index}">
          ${preview}
          <div class="model-card-body">
            <h4 class="model-card-title">${escapeHtml(item.title || item.name || item.id || "-")}</h4>
            <div class="model-card-id">${escapeHtml(item.id || "-")}</div>
            <div class="model-meta-line">${escapeHtml(t("modelSource"))}: ${escapeHtml(item.source || "unknown")} | ${escapeHtml(t("modelBase"))}: ${escapeHtml(
              item.base_model || "n/a",
            )}</div>
            <div class="model-meta-line">${escapeHtml(t("modelDownloads"))}: ${escapeHtml(item.downloads ?? "n/a")} | ${escapeHtml(
              t("modelLikes"),
            )}: ${escapeHtml(item.likes ?? "n/a")} | ${escapeHtml(t("modelSize"))}: ${escapeHtml(formatModelSize(item.size_bytes))}</div>
            <div>${statusBadge}</div>
            <div class="model-card-actions">
              <a href="${escapeHtml(item.model_url || "#")}" target="_blank" rel="noopener noreferrer">${escapeHtml(t("msgOpen"))}</a>
              <button type="button" class="search-detail-btn" data-index="${index}">${escapeHtml(t("msgDetail"))}</button>
              <button type="button" class="search-download-btn" data-index="${index}" ${!supportsDownload || installed ? "disabled" : ""}>${escapeHtml(
                t("btnDownload"),
              )}</button>
              <button type="button" class="search-apply-btn" data-index="${index}">${escapeHtml(t("msgApply"))}</button>
            </div>
          </div>
        </article>
      `;
    })
    .join("");

  cards.querySelectorAll(".search-detail-btn").forEach((button) => {
    button.addEventListener("click", async () => {
      const idx = Number(button.dataset.index || "-1");
      if (!Number.isInteger(idx) || idx < 0 || idx >= items.length) return;
      await openModelDetail(items[idx]);
    });
  });
  cards.querySelectorAll(".search-download-btn").forEach((button) => {
    button.addEventListener("click", async () => {
      const idx = Number(button.dataset.index || "-1");
      if (!Number.isInteger(idx) || idx < 0 || idx >= items.length) return;
      const item = items[idx];
      await startModelDownload(item.id, selectedDetailDownloadOptions(item));
    });
  });
  cards.querySelectorAll(".search-apply-btn").forEach((button) => {
    button.addEventListener("click", async () => {
      const idx = Number(button.dataset.index || "-1");
      if (!Number.isInteger(idx) || idx < 0 || idx >= items.length) return;
      await applySearchResultModel(items[idx]);
    });
  });
}

async function searchModels(event, options = {}) {
  if (event) event.preventDefault();
  const resetPage = options.resetPage !== false;
  if (resetPage) {
    state.searchPage = 1;
  } else if (Number.isInteger(options.page) && options.page > 0) {
    state.searchPage = options.page;
  }
  const rawLimit = Number(el("searchLimit").value || "30");
  const limit = Math.min(100, Math.max(1, Number.isFinite(rawLimit) ? Math.floor(rawLimit) : 30));
  el("searchLimit").value = String(limit);
  const baseModel = (el("searchBaseModel")?.value || "all").trim();
  const params = new URLSearchParams({
    task: el("searchTask").value,
    source: el("searchSource").value || "all",
    query: el("searchQuery").value.trim(),
    limit: String(limit),
    page: String(state.searchPage),
    sort: el("searchSort")?.value || "downloads",
    nsfw: el("searchNsfw")?.value || "exclude",
    model_kind: (el("searchModelKind")?.value || "").trim(),
  });
  if (baseModel && baseModel !== "all") {
    params.set("base_model", baseModel);
  }
  const data = await api(`/api/models/search2?${params.toString()}`);
  state.lastSearchResults = data.items || [];
  state.searchNextCursor = data.next_cursor || null;
  state.searchPrevCursor = data.prev_cursor || null;
  state.searchPage = Number(data.page_info?.page || state.searchPage || 1);
  state.searchDetail = null;
  const detailContent = el("modelDetailContent");
  const detailEmpty = el("modelDetailEmpty");
  if (detailContent) {
    detailContent.classList.remove("active");
    detailContent.innerHTML = "";
  }
  if (detailEmpty) {
    detailEmpty.style.display = "block";
    detailEmpty.textContent = t("msgModelDetailEmpty");
  }
  renderSearchResults(state.lastSearchResults);
  renderSearchPagination(data.page_info || null);
  const note = el("searchFilterNote");
  if (note) {
    note.textContent = `${t("modelSource")}: ${params.get("source")} | ${t("labelSearchSort")}: ${params.get("sort")} | ${t("labelLimit")}: ${limit}`;
  }
}

async function startModelDownload(repoId, extra = {}) {
  const targetDir = el("downloadTargetDir").value.trim();
  const data = await api("/api/models/download", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      repo_id: repoId,
      source: extra.source || null,
      hf_revision: extra.hf_revision || null,
      civitai_model_id: extra.civitai_model_id || null,
      civitai_version_id: extra.civitai_version_id || null,
      civitai_file_id: extra.civitai_file_id || null,
      target_dir: targetDir || null,
    }),
  });
  showTaskMessage(
    t("msgModelDownloadStarted", {
      repo: repoId,
      path: targetDir || t("msgDefaultModelsDir"),
    }),
  );
  trackTask(data.task_id);
}

async function generateText2Video(event) {
  event.preventDefault();
  const selectedModel = el("t2vModelSelect").value.trim();
  const loraIds = getSelectedValues("t2vLoraSelect");
  const payload = {
    prompt: el("t2vPrompt").value.trim(),
    negative_prompt: el("t2vNegative").value.trim(),
    model_id: selectedModel || null,
    lora_id: loraIds[0] || null,
    lora_ids: loraIds,
    lora_scale: Number(el("t2vLoraScale").value),
    backend: (el("t2vBackendSelect")?.value || "auto").trim(),
    num_inference_steps: Number(el("t2vSteps").value),
    num_frames: Number(el("t2vFrames").value),
    guidance_scale: Number(el("t2vGuidance").value),
    fps: Number(el("t2vFps").value),
    seed: readNum("t2vSeed"),
  };
  const data = await api("/api/generate/text2video", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  showTaskMessage(t("msgTextGenerationStarted", { id: data.task_id }));
  trackTask(data.task_id);
}

async function generateImage2Video(event) {
  event.preventDefault();
  const imageFile = el("i2vImage").files[0];
  if (!imageFile) throw new Error(t("msgInputImageRequired"));
  const formData = new FormData();
  const loraIds = getSelectedValues("i2vLoraSelect");
  formData.append("image", imageFile);
  formData.append("prompt", el("i2vPrompt").value.trim());
  formData.append("negative_prompt", el("i2vNegative").value.trim());
  formData.append("model_id", el("i2vModelSelect").value.trim());
  formData.append("lora_id", loraIds[0] || "");
  loraIds.forEach((value) => formData.append("lora_ids", value));
  formData.append("lora_scale", String(Number(el("i2vLoraScale").value)));
  formData.append("num_inference_steps", String(Number(el("i2vSteps").value)));
  formData.append("num_frames", String(Number(el("i2vFrames").value)));
  formData.append("guidance_scale", String(Number(el("i2vGuidance").value)));
  formData.append("fps", String(Number(el("i2vFps").value)));
  formData.append("width", String(Number(el("i2vWidth").value)));
  formData.append("height", String(Number(el("i2vHeight").value)));
  if (el("i2vSeed").value.trim()) formData.append("seed", el("i2vSeed").value.trim());

  const data = await api("/api/generate/image2video", {
    method: "POST",
    body: formData,
  });
  showTaskMessage(t("msgImageGenerationStarted", { id: data.task_id }));
  trackTask(data.task_id);
}

async function generateText2Image(event) {
  event.preventDefault();
  const selectedModel = el("t2iModelSelect").value.trim();
  const loraIds = getSelectedValues("t2iLoraSelect");
  const payload = {
    prompt: el("t2iPrompt").value.trim(),
    negative_prompt: el("t2iNegative").value.trim(),
    model_id: selectedModel || null,
    lora_id: loraIds[0] || null,
    lora_ids: loraIds,
    lora_scale: Number(el("t2iLoraScale").value),
    vae_id: el("t2iVaeSelect").value.trim() || null,
    num_inference_steps: Number(el("t2iSteps").value),
    guidance_scale: Number(el("t2iGuidance").value),
    width: Number(el("t2iWidth").value),
    height: Number(el("t2iHeight").value),
    seed: readNum("t2iSeed"),
  };
  const data = await api("/api/generate/text2image", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  showTaskMessage(t("msgTextImageGenerationStarted", { id: data.task_id }));
  trackTask(data.task_id);
}

async function generateImage2Image(event) {
  event.preventDefault();
  const imageFile = el("i2iImage").files[0];
  if (!imageFile) throw new Error(t("msgInputImageRequired"));
  const formData = new FormData();
  const loraIds = getSelectedValues("i2iLoraSelect");
  formData.append("image", imageFile);
  formData.append("prompt", el("i2iPrompt").value.trim());
  formData.append("negative_prompt", el("i2iNegative").value.trim());
  formData.append("model_id", el("i2iModelSelect").value.trim());
  formData.append("lora_id", loraIds[0] || "");
  loraIds.forEach((value) => formData.append("lora_ids", value));
  formData.append("lora_scale", String(Number(el("i2iLoraScale").value)));
  formData.append("vae_id", el("i2iVaeSelect").value.trim());
  formData.append("num_inference_steps", String(Number(el("i2iSteps").value)));
  formData.append("guidance_scale", String(Number(el("i2iGuidance").value)));
  formData.append("strength", String(Number(el("i2iStrength").value)));
  formData.append("width", String(Number(el("i2iWidth").value)));
  formData.append("height", String(Number(el("i2iHeight").value)));
  if (el("i2iSeed").value.trim()) formData.append("seed", el("i2iSeed").value.trim());
  const data = await api("/api/generate/image2image", {
    method: "POST",
    body: formData,
  });
  showTaskMessage(t("msgImageImageGenerationStarted", { id: data.task_id }));
  trackTask(data.task_id);
}

function renderTask(task) {
  renderTaskProgress(task);
  const base = t("taskLine", {
    id: task.id,
    type: translateTaskType(task.task_type),
    status: translateTaskStatus(task.status),
    progress: Math.round(taskProgressValue(task) * 100),
    message: translateServerMessage(task.message || ""),
  });
  if (task.error) {
    showTaskMessage(`${base} | ${t("taskError", { error: task.error })}`);
  } else {
    showTaskMessage(base);
  }
  const video = el("preview");
  const image = el("imagePreview");
  if (task.status === "completed" && task.result?.video_file) {
    image.style.display = "none";
    image.removeAttribute("src");
    video.src = `/api/videos/${encodeURIComponent(task.result.video_file)}?t=${Date.now()}`;
    video.style.display = "block";
  } else if (task.status === "completed" && task.result?.image_file) {
    video.style.display = "none";
    video.removeAttribute("src");
    image.src = `/api/images/${encodeURIComponent(task.result.image_file)}?t=${Date.now()}`;
    image.style.display = "block";
  }
}

function stopPolling() {
  if (state.pollTimer) {
    clearInterval(state.pollTimer);
    state.pollTimer = null;
  }
}

async function pollTask() {
  if (!state.currentTaskId) return;
  try {
    const task = await api(`/api/tasks/${state.currentTaskId}`);
    renderTask(task);
    if (task.status === "completed" || task.status === "error") {
      stopPolling();
      if (task.task_type === "download" && task.status === "completed") {
        await loadLocalModels();
      }
      if (task.status === "completed" && task.task_type !== "download") {
        try {
          await loadOutputs();
        } catch (error) {
          showTaskMessage(t("msgOutputsRefreshFailed", { error: error.message }));
        }
      }
    }
  } catch (error) {
    stopPolling();
    if (String(error.message).includes("404")) {
      saveLastTaskId(null);
      state.currentTaskId = null;
      renderTaskProgress(null);
    }
    showTaskMessage(t("msgTaskPollFailed", { error: error.message }));
  }
}

function trackTask(taskId) {
  state.currentTaskId = taskId;
  saveLastTaskId(taskId);
  stopPolling();
  pollTask();
  state.pollTimer = setInterval(pollTask, TASK_POLL_INTERVAL_MS);
}

async function restoreLastTask() {
  const lastTaskId = localStorage.getItem(TASK_STORAGE_KEY);
  if (!lastTaskId) return;
  try {
    const task = await api(`/api/tasks/${lastTaskId}`);
    state.currentTaskId = lastTaskId;
    renderTask(task);
    if (task.status === "queued" || task.status === "running") {
      stopPolling();
      state.pollTimer = setInterval(pollTask, TASK_POLL_INTERVAL_MS);
    }
  } catch (error) {
    saveLastTaskId(null);
  }
}

function bindModelSelectors() {
  if (el("t2iModelSelect")) {
    el("t2iModelSelect").addEventListener("change", async () => {
      renderModelPreview("text-to-image");
      try {
        await loadLoraCatalog("text-to-image", false);
        await loadVaeCatalog("text-to-image", false);
      } catch (error) {
        showTaskMessage(t("msgSearchFailed", { error: error.message }));
      }
    });
  }
  if (el("i2iModelSelect")) {
    el("i2iModelSelect").addEventListener("change", async () => {
      renderModelPreview("image-to-image");
      try {
        await loadLoraCatalog("image-to-image", false);
        await loadVaeCatalog("image-to-image", false);
      } catch (error) {
        showTaskMessage(t("msgSearchFailed", { error: error.message }));
      }
    });
  }
  if (el("t2vModelSelect")) {
    el("t2vModelSelect").addEventListener("change", async () => {
      renderModelPreview("text-to-video");
      try {
        await loadLoraCatalog("text-to-video", false);
      } catch (error) {
        showTaskMessage(t("msgSearchFailed", { error: error.message }));
      }
    });
  }
  if (el("i2vModelSelect")) {
    el("i2vModelSelect").addEventListener("change", async () => {
      renderModelPreview("image-to-video");
      try {
        await loadLoraCatalog("image-to-video", false);
      } catch (error) {
        showTaskMessage(t("msgSearchFailed", { error: error.message }));
      }
    });
  }
  bindClick("refreshT2IModels", async () => {
    try {
      await loadModelCatalog("text-to-image", true);
    } catch (error) {
      showTaskMessage(t("msgSearchFailed", { error: error.message }));
    }
  });
  bindClick("refreshI2IModels", async () => {
    try {
      await loadModelCatalog("image-to-image", true);
    } catch (error) {
      showTaskMessage(t("msgSearchFailed", { error: error.message }));
    }
  });
  bindClick("refreshT2VModels", async () => {
    try {
      await loadModelCatalog("text-to-video", true);
    } catch (error) {
      showTaskMessage(t("msgSearchFailed", { error: error.message }));
    }
  });
  bindClick("refreshI2VModels", async () => {
    try {
      await loadModelCatalog("image-to-video", true);
    } catch (error) {
      showTaskMessage(t("msgSearchFailed", { error: error.message }));
    }
  });
  bindClick("refreshT2ILoras", async () => {
    try {
      await loadLoraCatalog("text-to-image", true);
    } catch (error) {
      showTaskMessage(t("msgSearchFailed", { error: error.message }));
    }
  });
  bindClick("refreshI2ILoras", async () => {
    try {
      await loadLoraCatalog("image-to-image", true);
    } catch (error) {
      showTaskMessage(t("msgSearchFailed", { error: error.message }));
    }
  });
  bindClick("refreshT2VLoras", async () => {
    try {
      await loadLoraCatalog("text-to-video", true);
    } catch (error) {
      showTaskMessage(t("msgSearchFailed", { error: error.message }));
    }
  });
  bindClick("refreshI2VLoras", async () => {
    try {
      await loadLoraCatalog("image-to-video", true);
    } catch (error) {
      showTaskMessage(t("msgSearchFailed", { error: error.message }));
    }
  });
  bindClick("refreshT2IVaes", async () => {
    try {
      await loadVaeCatalog("text-to-image", true);
    } catch (error) {
      showTaskMessage(t("msgSearchFailed", { error: error.message }));
    }
  });
  bindClick("refreshI2IVaes", async () => {
    try {
      await loadVaeCatalog("image-to-image", true);
    } catch (error) {
      showTaskMessage(t("msgSearchFailed", { error: error.message }));
    }
  });
}

function bindLanguageSelector() {
  if (!el("languageSelect")) return;
  el("languageSelect").addEventListener("change", (event) => {
    setLanguage(event.target.value);
  });
}

async function bootstrap() {
  state.language = detectInitialLanguage();
  bindLanguageSelector();
  setLanguage(state.language);
  setTabs();
  bindModelSelectors();

  el("settingsForm").addEventListener("submit", async (event) => {
    try {
      await saveSettings(event);
    } catch (error) {
      showTaskMessage(t("msgSaveSettingsFailed", { error: error.message }));
    }
  });
  bindClick("clearHfCacheBtn", async () => {
    try {
      await clearHfCache();
    } catch (error) {
      showTaskMessage(t("msgHfCacheClearFailed", { error: error.message }));
    }
  });
  el("searchForm").addEventListener("submit", async (event) => {
    try {
      await searchModels(event);
    } catch (error) {
      showTaskMessage(t("msgSearchFailed", { error: error.message }));
    }
  });
  el("text2videoForm").addEventListener("submit", async (event) => {
    try {
      await generateText2Video(event);
    } catch (error) {
      showTaskMessage(t("msgTextGenerationFailed", { error: error.message }));
    }
  });
  el("image2videoForm").addEventListener("submit", async (event) => {
    try {
      await generateImage2Video(event);
    } catch (error) {
      showTaskMessage(t("msgImageGenerationFailed", { error: error.message }));
    }
  });
  el("image2imageForm").addEventListener("submit", async (event) => {
    try {
      await generateImage2Image(event);
    } catch (error) {
      showTaskMessage(t("msgImageGenerationFailed", { error: error.message }));
    }
  });
  el("text2imageForm").addEventListener("submit", async (event) => {
    try {
      await generateText2Image(event);
    } catch (error) {
      showTaskMessage(t("msgTextGenerationFailed", { error: error.message }));
    }
  });
  el("searchTask").addEventListener("change", async () => {
    refreshSearchSourceOptions();
    renderSearchBaseModelOptions();
    try {
      await searchModels(null, { resetPage: true });
    } catch (error) {
      showTaskMessage(t("msgSearchFailed", { error: error.message }));
    }
  });
  el("searchSource").addEventListener("change", async () => {
    try {
      await searchModels(null, { resetPage: true });
    } catch (error) {
      showTaskMessage(t("msgSearchFailed", { error: error.message }));
    }
  });
  el("searchBaseModel").addEventListener("change", async () => {
    try {
      await searchModels(null, { resetPage: true });
    } catch (error) {
      showTaskMessage(t("msgSearchFailed", { error: error.message }));
    }
  });
  el("searchSort").addEventListener("change", async () => {
    try {
      await searchModels(null, { resetPage: true });
    } catch (error) {
      showTaskMessage(t("msgSearchFailed", { error: error.message }));
    }
  });
  el("searchNsfw").addEventListener("change", async () => {
    try {
      await searchModels(null, { resetPage: true });
    } catch (error) {
      showTaskMessage(t("msgSearchFailed", { error: error.message }));
    }
  });
  el("searchModelKind").addEventListener("change", async () => {
    try {
      await searchModels(null, { resetPage: true });
    } catch (error) {
      showTaskMessage(t("msgSearchFailed", { error: error.message }));
    }
  });
  el("searchViewMode").addEventListener("change", () => {
    renderSearchResults(state.lastSearchResults || []);
  });
  el("searchPrevBtn").addEventListener("click", async () => {
    const page = Math.max(1, Number(state.searchPage || 1) - 1);
    try {
      await searchModels(null, { resetPage: false, page });
    } catch (error) {
      showTaskMessage(t("msgSearchFailed", { error: error.message }));
    }
  });
  el("searchNextBtn").addEventListener("click", async () => {
    const page = Math.max(1, Number(state.searchPage || 1) + 1);
    try {
      await searchModels(null, { resetPage: false, page });
    } catch (error) {
      showTaskMessage(t("msgSearchFailed", { error: error.message }));
    }
  });
  el("refreshLocalModels").addEventListener("click", async () => {
    try {
      await loadLocalModels();
    } catch (error) {
      showTaskMessage(t("msgLocalModelRefreshFailed", { error: error.message }));
    }
  });
  el("refreshOutputs").addEventListener("click", async () => {
    try {
      await loadOutputs();
    } catch (error) {
      showTaskMessage(t("msgOutputsRefreshFailed", { error: error.message }));
    }
  });
  el("localModelsDir").addEventListener("change", async () => {
    try {
      await loadLocalModels();
    } catch (error) {
      showTaskMessage(t("msgLocalModelRefreshFailed", { error: error.message }));
    }
  });
  el("localLineageFilter").addEventListener("change", () => {
    state.localLineageFilter = el("localLineageFilter").value || "all";
    renderLocalModels(state.localModels || [], state.localModelsBaseDir || "");
  });
  el("cfgModelsDir").addEventListener("change", async () => {
    try {
      await loadSettingsLocalModels();
    } catch (error) {
      showTaskMessage(t("msgLocalModelRefreshFailed", { error: error.message }));
    }
  });
  try {
    await Promise.all([loadRuntimeInfo(), loadSettings()]);
    await Promise.all([loadLocalModels(), loadOutputs()]);
    await Promise.all([
      loadModelCatalog("text-to-image", false),
      loadModelCatalog("image-to-image", false),
      loadModelCatalog("text-to-video", false),
      loadModelCatalog("image-to-video", false),
    ]);
    await searchModels(null, { resetPage: true });
    await restoreLastTask();
  } catch (error) {
    showTaskMessage(t("msgInitFailed", { error: error.message }));
  }
}

bootstrap();

===== FILE: README.md =====

# ROCm VideoGen Web App

ROCm 環境で動作する `Text-to-Image` / `Image-to-Image` / `Text-to-Video` / `Image-to-Video` の Web アプリです。

## 機能

- Web GUI からテキスト→動画生成
- Web GUI からテキスト→画像生成
- Web GUI から画像→画像生成
- Web GUI から画像→動画生成
- 各生成機能で複数LoRAを選択して適用可能（共通スケール）
- Text-to-Image / Image-to-Image で VAE 選択可能（ローカルVAEカタログ）
- Hugging Face モデル検索
- CivitAI モデル検索（画像系タスク）
- Model Search のカードUI（Grid/List）、詳細ペイン、ページング
- Model Search の詳細表示（説明、タグ、プレビュー、バージョン/ファイル選択）
- Hugging Face / CivitAI のモデルダウンロード
- 検索語なしでもモデル検索可能（タスク別の人気順）
- 検索結果にプレビュー画像がある場合は表示
- 生成モデルはドロップダウンから選択可能
- モデル選択時にサムネイル表示
- サーバサイドでモデルをダウンロード
- `Models` タブからローカルモデルを削除
- 設定値保存 (`data/settings.json`)
- 生成/ダウンロード失敗時の詳細ログ出力（スタックトレース含む）
- 生成ジョブの進捗表示と動画/画像プレビュー
- Web GUI 多言語表示（`en`, `ja`, `es`, `fr`, `de`, `it`, `pt`, `ru`, `ar`）
- モデル保存先をフルパスで指定可能
- ローカルモデル一覧にサムネイル表示、系譜ドロップダウン絞り込み、生成タスクへ適用ボタン
- リッスンポートを `Settings` 画面から変更可能（次回起動時に反映）

## 前提

- Windows + ROCm 対応 GPU
- Python 3.12 推奨
- ROCm 7.2 を使う場合は、以下のセットアップバッチを使用

## ROCm 7.2 セットアップ

```bash
setup_rocm72.bat
```

`setup_rocm72.bat` は以下を実行します。

- `venv` 作成（Python 3.12）
- ROCm 7.2 SDK wheel インストール
- ROCm 7.2 対応 `torch/torchaudio/torchvision` インストール
- `requirements.txt` インストール

## 起動

```bash
start.bat
```

`start.bat` は `venv\Scripts\python.exe` を優先して起動します。  
`venv` がない場合は `.venv` を作成して起動します。
サーバー起動を検知してからブラウザを自動起動します（無効化: `set AUTO_OPEN_BROWSER=0`）。
Hugging Face キャッシュは `Settings` 画面の `Clear Cache` ボタンから削除できます。

ブラウザで `http://localhost:8000` を開いて利用します。

## 使い方

1. `Settings` タブで保存先ディレクトリやデフォルトモデルを確認・保存
2. `Models` タブでモデルを検索し、`Download` を実行（Hugging Face / CivitAI をサーバ側に保存）
3. `Local Models` タブで系譜ドロップダウンで絞り込み、`Set T2I/I2I/T2V/I2V` を使って生成タブへ反映
4. `Text to Image` / `Image to Image` / `Text to Video` / `Image to Video` タブで生成ジョブを開始
5. 下部ステータスで進捗を確認し、完了後に画像または動画をプレビュー

LoRA は各生成タブで複数選択できます。`LoRA Scale` は選択したすべての LoRA に同じ値で適用されます。  
VAE は `Text to Image` / `Image to Image` タブで選択できます。

`Models` タブの `Download Save Path` に保存先パスを入れると、ダウンロードごとに保存先を選択できます。  
未入力の場合は `Settings` の `Models Directory` が使われます。

## Model Search API（拡張）

既存の `GET /api/models/search` は後方互換のまま維持されています。  
新UIは `search2/detail` を利用します。

- `GET /api/models/search2`
  - query:
    - `task`（必須）: `text-to-image|image-to-image|text-to-video|image-to-video`
    - `source`: `all|huggingface|civitai`
    - `query`: 検索語（空欄可）
    - `base_model`: ベースモデルフィルタ
    - `sort`: `popularity|downloads|likes|updated|created`
    - `nsfw`: `include|exclude`
    - `model_kind`: `checkpoint|lora|vae|controlnet|embedding|upscaler`（provider対応範囲内）
    - `limit`: 1-100
    - `page` または `cursor`
  - response:
    - `items[]`
    - `next_cursor`, `prev_cursor`
    - `page_info`

- `GET /api/models/detail`
  - query:
    - `source`: `huggingface|civitai`
    - `id`: HF repo id または CivitAI id（`civitai/123` 形式可）
  - response:
    - `title/name/id/source`
    - `description`
    - `tags[]`
    - `previews[]`
    - `versions[]`（`files[]` 含む）

- `POST /api/models/download`（後方互換拡張）
  - 従来:
    - `repo_id`, `revision`, `target_dir`
  - 追加:
    - `source`（`huggingface|civitai`）
    - `hf_revision`
    - `civitai_model_id`, `civitai_version_id`, `civitai_file_id`
  - 省略時は従来動作
  - ダウンロード後に `model_meta.json` を保存（CivitAIは `civitai_model.json` も保存）

## 手動テスト手順（Model Search）

1. `start.bat` で起動し、`Models` タブを開く
2. 検索条件を設定
   - `Source=all`, `Task=text-to-image`, `Limit=30`
   - 必要に応じて `Sort/NSFW/Model Kind/Base Model` を設定
3. `Search Models` を実行
   - カードが Grid/List で表示されること
   - `Prev/Next` でページ遷移できること
4. 任意カードの `Detail` を押す
   - 右ペインに説明、タグ、プレビュー、バージョン/ファイルが表示されること
5. ダウンロード
   - HF: revision（select/manual）を指定して `Download`
   - CivitAI: version/file を選択して `Download`
6. 完了後、`Local Models` を更新
   - ダウンロード済みモデルが表示されること
7. `Apply` を押す
   - 選択中 task のモデル選択へ反映されること

## 保存先

- 設定: `data/settings.json`
  - `server.rocm_aotriton_experimental`: `true/false`（`start.bat` 起動時に `TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1/0` へ反映）
- モデル: `models/`
- 生成画像/動画: `outputs/`
- 一時画像: `tmp/`
- ログ: `logs/YYYYMMDD_HHMMSS_videogen_pid<process-id>.log`（起動プロセスごとに作成、`data/settings.json` の `paths.logs_dir` で変更可）

## ログ確認

- 直近ログAPI: `GET /api/logs/recent?lines=200`
- 失敗時はタスク詳細の `error` と、ログの `Traceback` / `diagnostics` を確認してください。

## 注意

- 初回モデル読み込みは時間がかかります。
- VRAM 使用量が大きいため、`frames` や `steps` を調整してください。
- 入力モデルIDは Hugging Face repo ID またはローカルパスを使用できます。

===== FILE: tests/integration/test_api.py =====

import json
import time
from pathlib import Path

import pytest
import numpy as np
from fastapi.testclient import TestClient
from PIL import Image

import main

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def client() -> TestClient:
    return TestClient(main.app)


@pytest.fixture(autouse=True, scope="module")
def isolate_settings_store(tmp_path_factory: pytest.TempPathFactory):
    original_store = main.settings_store
    settings_path = tmp_path_factory.mktemp("settings") / "settings.json"
    main.settings_store = main.SettingsStore(settings_path, main.DEFAULT_SETTINGS)
    main.ensure_runtime_dirs(main.settings_store.get())
    yield
    main.settings_store = original_store
    main.ensure_runtime_dirs(main.settings_store.get())


def test_settings_roundtrip(client: TestClient) -> None:
    got = client.get("/api/settings")
    assert got.status_code == 200
    payload = got.json()
    payload["paths"]["tmp_dir"] = "tmp"
    payload["defaults"]["fps"] = 12

    updated = client.put("/api/settings", json=payload)
    assert updated.status_code == 200
    assert updated.json()["defaults"]["fps"] == 12

    verify = client.get("/api/settings")
    assert verify.status_code == 200
    assert verify.json()["defaults"]["fps"] == 12


def test_models_local_with_custom_dir(client: TestClient, tmp_path: Path) -> None:
    custom = tmp_path / "my-models"
    (custom / "org--model").mkdir(parents=True)
    resp = client.get("/api/models/local", params={"dir": str(custom)})
    assert resp.status_code == 200
    body = resp.json()
    assert body["base_dir"] == str(custom.resolve())
    assert any(item["name"] == "org--model" for item in body["items"])


def test_delete_local_model_endpoint(client: TestClient, tmp_path: Path) -> None:
    base = tmp_path / "delete-target"
    model_dir = base / "org--sample"
    (model_dir / ".cache" / "huggingface").mkdir(parents=True)
    (model_dir / "weights.safetensors").write_bytes(b"123")

    listing = client.get("/api/models/local", params={"dir": str(base)})
    assert listing.status_code == 200
    item = next(x for x in listing.json()["items"] if x["name"] == "org--sample")
    assert item["can_delete"] is True

    resp = client.post("/api/models/local/delete", json={"model_name": "org--sample", "base_dir": str(base)})
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"
    assert not model_dir.exists()


def test_delete_local_model_rejects_non_model_directory(client: TestClient, tmp_path: Path) -> None:
    base = tmp_path / "delete-invalid"
    target = base / "random-dir"
    target.mkdir(parents=True)
    (target / "note.txt").write_text("hello", encoding="utf-8")

    resp = client.post("/api/models/local/delete", json={"model_name": "random-dir", "base_dir": str(base)})
    assert resp.status_code == 400


def test_outputs_list_and_delete(client: TestClient, tmp_path: Path) -> None:
    outputs_dir = tmp_path / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (64, 64), color=(1, 2, 3)).save(outputs_dir / "sample.png")
    (outputs_dir / "sample.mp4").write_bytes(b"\x00\x00\x00\x18ftypmp42")

    settings = client.get("/api/settings").json()
    settings["paths"]["outputs_dir"] = str(outputs_dir.resolve())
    put_resp = client.put("/api/settings", json=settings)
    assert put_resp.status_code == 200

    listing = client.get("/api/outputs", params={"limit": 50})
    assert listing.status_code == 200
    body = listing.json()
    assert body["base_dir"] == str(outputs_dir.resolve())
    items = body["items"]
    names = {item["name"] for item in items}
    assert "sample.png" in names
    assert "sample.mp4" in names
    image_item = next(item for item in items if item["name"] == "sample.png")
    assert image_item["kind"] == "image"
    assert image_item["view_url"].startswith("/api/images/")
    video_item = next(item for item in items if item["name"] == "sample.mp4")
    assert video_item["kind"] == "video"
    assert video_item["view_url"].startswith("/api/videos/")

    delete_resp = client.post("/api/outputs/delete", json={"file_name": "sample.png"})
    assert delete_resp.status_code == 200
    assert not (outputs_dir / "sample.png").exists()


def test_outputs_delete_rejects_invalid_name(client: TestClient) -> None:
    resp = client.post("/api/outputs/delete", json={"file_name": "../escape.txt"})
    assert resp.status_code == 400


def test_download_task_lifecycle(client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_snapshot_download(repo_id: str, local_dir: str, **kwargs):
        path = Path(local_dir)
        path.mkdir(parents=True, exist_ok=True)
        (path / "model_index.json").write_text(json.dumps({"repo_id": repo_id}), encoding="utf-8")
        return str(path)

    monkeypatch.setattr(main, "snapshot_download", fake_snapshot_download)
    target_dir = tmp_path / "download-target"
    post = client.post("/api/models/download", json={"repo_id": "foo/bar", "target_dir": str(target_dir)})
    assert post.status_code == 200
    task_id = post.json()["task_id"]

    deadline = time.time() + 5
    last = None
    while time.time() < deadline:
        status = client.get(f"/api/tasks/{task_id}")
        assert status.status_code == 200
        last = status.json()
        if last["status"] in ("completed", "error"):
            break
        time.sleep(0.1)

    assert last is not None
    assert last["status"] == "completed"
    assert Path(last["result"]["local_path"]).exists()


def test_download_task_lifecycle_civitai(client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeResponse:
        def __init__(self, payload: bytes, content_length: int | None = None) -> None:
            self._payload = payload
            self._offset = 0
            self.headers = {"Content-Length": str(content_length)} if content_length is not None else {}

        def read(self, size: int = -1) -> bytes:
            if size is None or size < 0:
                data = self._payload[self._offset :]
                self._offset = len(self._payload)
                return data
            if self._offset >= len(self._payload):
                return b""
            end = min(self._offset + size, len(self._payload))
            data = self._payload[self._offset : end]
            self._offset = end
            return data

        def close(self) -> None:
            return None

    model_payload = {
        "id": 123,
        "name": "Civit Model",
        "modelVersions": [
            {
                "id": 456,
                "name": "v1",
                "files": [
                    {
                        "id": 789,
                        "name": "model.safetensors",
                        "type": "Model",
                        "downloadUrl": "https://example.invalid/model.safetensors",
                    }
                ],
            }
        ],
    }
    model_bytes = b"abcdef"

    def fake_urlopen(request_obj, timeout: int = 20):
        url = request_obj.full_url
        if "/api/v1/models/123" in url:
            return FakeResponse(json.dumps(model_payload).encode("utf-8"))
        if "example.invalid/model.safetensors" in url:
            return FakeResponse(model_bytes, content_length=len(model_bytes))
        raise AssertionError(f"unexpected url: {url}")

    monkeypatch.setattr(main, "urlopen", fake_urlopen)

    target_dir = tmp_path / "download-target-civitai"
    post = client.post("/api/models/download", json={"repo_id": "civitai/123", "target_dir": str(target_dir)})
    assert post.status_code == 200
    task_id = post.json()["task_id"]

    deadline = time.time() + 5
    last = None
    while time.time() < deadline:
        status = client.get(f"/api/tasks/{task_id}")
        assert status.status_code == 200
        last = status.json()
        if last["status"] in ("completed", "error"):
            break
        time.sleep(0.1)

    assert last is not None
    assert last["status"] == "completed"
    model_dir = Path(last["result"]["local_path"])
    assert model_dir.exists()
    assert (model_dir / "model.safetensors").read_bytes() == model_bytes
    metadata = json.loads((model_dir / "civitai_model.json").read_text(encoding="utf-8"))
    assert metadata["model_id"] == 123


def test_model_catalog_endpoint(client: TestClient) -> None:
    resp = client.get("/api/models/catalog", params={"task": "text-to-video", "limit": 5})
    assert resp.status_code == 200
    body = resp.json()
    assert "items" in body
    assert "default_model" in body


def test_models_search_merges_hf_and_civitai(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        main,
        "search_hf_models",
        lambda task, query, limit, token: [
            {"id": "hf/a", "pipeline_tag": task, "downloads": 100, "likes": 10, "size_bytes": 1, "source": "huggingface", "download_supported": True}
        ],
    )
    monkeypatch.setattr(
        main,
        "search_civitai_models",
        lambda task, query, limit: [
            {"id": "civitai/1", "pipeline_tag": task, "downloads": 90, "likes": 9, "size_bytes": 2, "source": "civitai", "download_supported": True}
        ],
    )
    resp = client.get("/api/models/search", params={"task": "text-to-image", "query": "test", "limit": 6})
    assert resp.status_code == 200
    ids = [item["id"] for item in resp.json()["items"]]
    assert "hf/a" in ids
    assert "civitai/1" in ids


def test_models_search2_returns_page_info_and_installed(client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    models_dir = tmp_path / "models_search2"
    (models_dir / "hf--a").mkdir(parents=True)
    settings = client.get("/api/settings").json()
    settings["paths"]["models_dir"] = str(models_dir.resolve())
    put_resp = client.put("/api/settings", json=settings)
    assert put_resp.status_code == 200

    monkeypatch.setattr(
        main,
        "search_hf_models_v2",
        lambda **_kwargs: {
            "items": [
                {
                    "id": "hf/a",
                    "title": "hf/a",
                    "pipeline_tag": "text-to-image",
                    "downloads": 1,
                    "likes": 2,
                    "size_bytes": 3,
                    "source": "huggingface",
                    "download_supported": True,
                    "base_model": "StableDiffusion XL",
                }
            ],
            "has_next": True,
        },
    )
    monkeypatch.setattr(main, "search_civitai_models_v2", lambda **_kwargs: {"items": [], "has_next": False})

    resp = client.get(
        "/api/models/search2",
        params={"task": "text-to-image", "query": "", "limit": 30, "source": "huggingface", "page": 1},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["page_info"]["page"] == 1
    assert body["next_cursor"] == "2"
    assert body["items"][0]["installed"] is True


def test_models_detail_endpoints(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        main,
        "get_hf_model_detail",
        lambda repo_id, token: {"source": "huggingface", "id": repo_id, "title": repo_id, "versions": [], "previews": [], "tags": []},
    )
    monkeypatch.setattr(
        main,
        "get_civitai_model_detail",
        lambda model_id: {"source": "civitai", "id": f"civitai/{model_id}", "title": "x", "versions": [], "previews": [], "tags": []},
    )

    hf = client.get("/api/models/detail", params={"source": "huggingface", "id": "foo/bar"})
    assert hf.status_code == 200
    assert hf.json()["id"] == "foo/bar"

    civitai = client.get("/api/models/detail", params={"source": "civitai", "id": "civitai/123"})
    assert civitai.status_code == 200
    assert civitai.json()["id"] == "civitai/123"


def test_models_search_source_filter(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    called = {"hf": 0, "civitai": 0}

    def fake_hf(task, query, limit, token):
        called["hf"] += 1
        return [
            {"id": "hf/only", "pipeline_tag": task, "downloads": 10, "likes": 1, "size_bytes": 1, "source": "huggingface", "download_supported": True}
        ]

    def fake_civitai(task, query, limit):
        called["civitai"] += 1
        return [
            {"id": "civitai/only", "pipeline_tag": task, "downloads": 9, "likes": 1, "size_bytes": 1, "source": "civitai", "download_supported": True}
        ]

    monkeypatch.setattr(main, "search_hf_models", fake_hf)
    monkeypatch.setattr(main, "search_civitai_models", fake_civitai)

    hf_resp = client.get("/api/models/search", params={"task": "text-to-image", "query": "", "limit": 6, "source": "huggingface"})
    assert hf_resp.status_code == 200
    hf_ids = [item["id"] for item in hf_resp.json()["items"]]
    assert hf_ids == ["hf/only"]

    civitai_resp = client.get("/api/models/search", params={"task": "text-to-image", "query": "", "limit": 6, "source": "civitai"})
    assert civitai_resp.status_code == 200
    civitai_ids = [item["id"] for item in civitai_resp.json()["items"]]
    assert civitai_ids == ["civitai/only"]

    assert called["hf"] == 1
    assert called["civitai"] == 1


def test_models_search_civitai_live_parser_path(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(main, "search_hf_models", lambda task, query, limit, token: [])

    class FakeResponse:
        def __init__(self, payload: dict) -> None:
            self._payload = payload

        def read(self) -> bytes:
            return json.dumps(self._payload).encode("utf-8")

        def close(self) -> None:
            return None

    payload = {
        "items": [
            {
                "id": 12345,
                "name": "Example CivitAI",
                "type": "Checkpoint",
                "stats": {"downloadCount": 1200, "favoriteCount": 12},
                    "modelVersions": [
                        {
                            "images": [{"url": "https://example.invalid/preview.jpg"}],
                            "files": [{"sizeKB": 2048, "downloadUrl": "https://example.invalid/model.safetensors"}],
                        }
                    ],
                }
            ]
    }

    def fake_urlopen(request_obj, timeout: int = 20):
        assert "civitai.com/api/v1/models" in request_obj.full_url
        assert timeout == 20
        return FakeResponse(payload)

    monkeypatch.setattr(main, "urlopen", fake_urlopen)

    resp = client.get("/api/models/search", params={"task": "text-to-image", "query": "", "limit": 6})
    assert resp.status_code == 200
    body = resp.json()
    civitai = next(item for item in body["items"] if item["id"] == "civitai/12345")
    assert civitai["source"] == "civitai"
    assert civitai["download_supported"] is True
    assert civitai["preview_url"] == "https://example.invalid/preview.jpg"
    assert civitai["size_bytes"] == 2048 * 1024


def test_download_task_lifecycle_civitai_with_selected_file(client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeResponse:
        def __init__(self, payload: bytes, content_length: int | None = None) -> None:
            self._payload = payload
            self._offset = 0
            self.headers = {"Content-Length": str(content_length)} if content_length is not None else {}

        def read(self, size: int = -1) -> bytes:
            if size is None or size < 0:
                data = self._payload[self._offset :]
                self._offset = len(self._payload)
                return data
            if self._offset >= len(self._payload):
                return b""
            end = min(self._offset + size, len(self._payload))
            data = self._payload[self._offset : end]
            self._offset = end
            return data

        def close(self) -> None:
            return None

    model_payload = {
        "id": 123,
        "name": "Civit Model",
        "modelVersions": [
            {
                "id": 456,
                "name": "v1",
                "images": [{"url": "https://example.invalid/preview.jpg"}],
                "files": [
                    {"id": 789, "name": "file_a.safetensors", "type": "Model", "downloadUrl": "https://example.invalid/file_a"}
                ],
            },
            {
                "id": 999,
                "name": "v2",
                "files": [
                    {"id": 555, "name": "file_b.safetensors", "type": "Model", "downloadUrl": "https://example.invalid/file_b"}
                ],
            },
        ],
    }

    def fake_urlopen(request_obj, timeout: int = 20):
        url = request_obj.full_url
        if "/api/v1/models/123" in url:
            return FakeResponse(json.dumps(model_payload).encode("utf-8"))
        if "example.invalid/file_b" in url:
            payload = b"selected"
            return FakeResponse(payload, content_length=len(payload))
        if "example.invalid/preview.jpg" in url:
            payload = b"img"
            return FakeResponse(payload, content_length=len(payload))
        raise AssertionError(f"unexpected url: {url}")

    monkeypatch.setattr(main, "urlopen", fake_urlopen)
    target_dir = tmp_path / "download-target-civitai-selected"
    post = client.post(
        "/api/models/download",
        json={
            "repo_id": "civitai/123",
            "source": "civitai",
            "civitai_model_id": 123,
            "civitai_version_id": 999,
            "civitai_file_id": 555,
            "target_dir": str(target_dir),
        },
    )
    assert post.status_code == 200
    task_id = post.json()["task_id"]

    deadline = time.time() + 5
    last = None
    while time.time() < deadline:
        status = client.get(f"/api/tasks/{task_id}")
        assert status.status_code == 200
        last = status.json()
        if last["status"] in ("completed", "error"):
            break
        time.sleep(0.1)

    assert last is not None
    assert last["status"] == "completed"
    model_dir = Path(last["result"]["local_path"])
    assert (model_dir / "file_b.safetensors").read_bytes() == b"selected"


def test_model_catalog_filters_incompatible_local_models(client: TestClient, tmp_path: Path) -> None:
    models_dir = tmp_path / "models"
    t2i = models_dir / "ok--text-image"
    i2i = models_dir / "ok--image-image"
    t2v = models_dir / "ok--text"
    i2v = models_dir / "ok--image"
    lora = models_dir / "bad--lora"
    t2i.mkdir(parents=True)
    i2i.mkdir(parents=True)
    t2v.mkdir(parents=True)
    i2v.mkdir(parents=True)
    lora.mkdir(parents=True)
    (t2i / "model_index.json").write_text(json.dumps({"_class_name": "StableDiffusionPipeline"}), encoding="utf-8")
    (i2i / "model_index.json").write_text(json.dumps({"_class_name": "StableDiffusionImg2ImgPipeline"}), encoding="utf-8")
    (t2v / "model_index.json").write_text(json.dumps({"_class_name": "TextToVideoSDPipeline"}), encoding="utf-8")
    (i2v / "model_index.json").write_text(json.dumps({"_class_name": "I2VGenXLPipeline"}), encoding="utf-8")
    (lora / "README.md").write_text("no pipeline", encoding="utf-8")

    settings = client.get("/api/settings").json()
    settings["paths"]["models_dir"] = str(models_dir.resolve())
    put_resp = client.put("/api/settings", json=settings)
    assert put_resp.status_code == 200

    t2i_resp = client.get("/api/models/catalog", params={"task": "text-to-image", "limit": 20})
    assert t2i_resp.status_code == 200
    t2i_ids = {item["id"] for item in t2i_resp.json()["items"]}
    assert "ok/text-image" in t2i_ids
    assert "ok/image-image" not in t2i_ids
    assert "ok/text" not in t2i_ids
    assert "ok/image" not in t2i_ids
    assert "bad/lora" not in t2i_ids

    i2i_resp = client.get("/api/models/catalog", params={"task": "image-to-image", "limit": 20})
    assert i2i_resp.status_code == 200
    i2i_ids = {item["id"] for item in i2i_resp.json()["items"]}
    assert "ok/text-image" in i2i_ids
    assert "ok/image-image" in i2i_ids
    assert "ok/text" not in i2i_ids
    assert "ok/image" not in i2i_ids
    assert "bad/lora" not in i2i_ids

    t2v_resp = client.get("/api/models/catalog", params={"task": "text-to-video", "limit": 20})
    assert t2v_resp.status_code == 200
    t2v_ids = {item["id"] for item in t2v_resp.json()["items"]}
    assert "ok/text" in t2v_ids
    assert "ok/image" not in t2v_ids
    assert "bad/lora" not in t2v_ids

    i2v_resp = client.get("/api/models/catalog", params={"task": "image-to-video", "limit": 20})
    assert i2v_resp.status_code == 200
    i2v_ids = {item["id"] for item in i2v_resp.json()["items"]}
    assert "ok/image" in i2v_ids
    assert "ok/text" not in i2v_ids
    assert "bad/lora" not in i2v_ids


def test_local_models_include_preview_and_meta(client: TestClient, tmp_path: Path) -> None:
    models_dir = tmp_path / "models_local_meta"
    model_dir = models_dir / "org--base-model"
    model_dir.mkdir(parents=True)
    (model_dir / "model_index.json").write_text(
        json.dumps({"_class_name": "StableDiffusionPipeline", "_name_or_path": "runwayml/stable-diffusion-v1-5"}),
        encoding="utf-8",
    )
    Image.new("RGB", (32, 32), color=(1, 2, 3)).save(model_dir / "thumbnail.png")
    lora_dir = models_dir / "org--style-lora"
    lora_dir.mkdir(parents=True)
    (lora_dir / "adapter_config.json").write_text(
        json.dumps({"base_model_name_or_path": "runwayml/stable-diffusion-v1-5"}),
        encoding="utf-8",
    )
    (lora_dir / "pytorch_lora_weights.safetensors").write_bytes(b"123")

    resp = client.get("/api/models/local", params={"dir": str(models_dir)})
    assert resp.status_code == 200
    items = resp.json()["items"]
    base_item = next(item for item in items if item["name"] == "org--base-model")
    assert base_item["preview_url"]
    assert "text-to-image" in base_item["compatible_tasks"]
    assert base_item["base_model"] == "runwayml/stable-diffusion-v1-5"
    lora_item = next(item for item in items if item["name"] == "org--style-lora")
    assert lora_item["is_lora"] is True
    assert lora_item["base_model"] == "runwayml/stable-diffusion-v1-5"


def test_text2image_task_lifecycle_and_image_endpoint(client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    outputs_dir = tmp_path / "outputs"
    models_dir = tmp_path / "models"
    tmp_dir = tmp_path / "tmp"
    settings = client.get("/api/settings").json()
    settings["paths"]["outputs_dir"] = str(outputs_dir.resolve())
    settings["paths"]["models_dir"] = str(models_dir.resolve())
    settings["paths"]["tmp_dir"] = str(tmp_dir.resolve())
    put_resp = client.put("/api/settings", json=settings)
    assert put_resp.status_code == 200

    class FakeText2ImagePipeline:
        def __call__(
            self,
            prompt: str,
            negative_prompt: str | None = None,
            num_inference_steps: int = 30,
            guidance_scale: float = 7.5,
            width: int = 512,
            height: int = 512,
            generator: object | None = None,
        ) -> object:
            image = Image.new("RGB", (width, height), color=(12, 34, 56))
            return type("FakeOut", (), {"images": [image]})()

    monkeypatch.setattr(main, "TORCH_IMPORT_ERROR", None)
    monkeypatch.setattr(main, "get_device_and_dtype", lambda: ("cpu", "float32"))
    monkeypatch.setattr(main, "get_pipeline", lambda kind, model_ref, settings_payload: FakeText2ImagePipeline())

    post = client.post(
        "/api/generate/text2image",
        json={
            "prompt": "test prompt",
            "negative_prompt": "",
            "model_id": "",
            "num_inference_steps": 4,
            "guidance_scale": 7.5,
            "width": 320,
            "height": 256,
        },
    )
    assert post.status_code == 200
    task_id = post.json()["task_id"]

    deadline = time.time() + 5
    last = None
    while time.time() < deadline:
        status = client.get(f"/api/tasks/{task_id}")
        assert status.status_code == 200
        last = status.json()
        if last["status"] in ("completed", "error"):
            break
        time.sleep(0.1)

    assert last is not None
    assert last["status"] == "completed"
    image_file = last["result"]["image_file"]
    image_path = outputs_dir / image_file
    assert image_path.exists()

    image_resp = client.get(f"/api/images/{image_file}")
    assert image_resp.status_code == 200
    assert image_resp.headers["content-type"].startswith("image/png")


def test_image2image_task_lifecycle(client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    outputs_dir = tmp_path / "outputs_i2i"
    models_dir = tmp_path / "models_i2i"
    tmp_dir = tmp_path / "tmp_i2i"
    settings = client.get("/api/settings").json()
    settings["paths"]["outputs_dir"] = str(outputs_dir.resolve())
    settings["paths"]["models_dir"] = str(models_dir.resolve())
    settings["paths"]["tmp_dir"] = str(tmp_dir.resolve())
    settings["defaults"]["image2image_model"] = "runwayml/stable-diffusion-v1-5"
    put_resp = client.put("/api/settings", json=settings)
    assert put_resp.status_code == 200

    class FakeImage2ImagePipeline:
        def __call__(
            self,
            prompt: str,
            image: Image.Image,
            negative_prompt: str | None = None,
            num_inference_steps: int = 30,
            guidance_scale: float = 7.5,
            strength: float = 0.6,
            generator: object | None = None,
        ) -> object:
            result = Image.new("RGB", image.size, color=(77, 99, 11))
            return type("FakeOut", (), {"images": [result]})()

    monkeypatch.setattr(main, "TORCH_IMPORT_ERROR", None)
    monkeypatch.setattr(main, "get_device_and_dtype", lambda: ("cpu", "float32"))
    monkeypatch.setattr(main, "get_pipeline", lambda kind, model_ref, settings_payload: FakeImage2ImagePipeline())

    input_image = tmp_path / "input.png"
    Image.new("RGB", (256, 128), color=(1, 2, 3)).save(input_image)
    with input_image.open("rb") as handle:
        post = client.post(
            "/api/generate/image2image",
            files={"image": ("input.png", handle, "image/png")},
            data={
                "prompt": "refine this image",
                "negative_prompt": "",
                "model_id": "",
                "num_inference_steps": "6",
                "guidance_scale": "7.5",
                "strength": "0.55",
                "width": "256",
                "height": "128",
            },
        )
    assert post.status_code == 200
    task_id = post.json()["task_id"]

    deadline = time.time() + 5
    last = None
    while time.time() < deadline:
        status = client.get(f"/api/tasks/{task_id}")
        assert status.status_code == 200
        last = status.json()
        if last["status"] in ("completed", "error"):
            break
        time.sleep(0.1)

    assert last is not None
    assert last["status"] == "completed"
    assert "image_file" in last["result"]


def test_lora_catalog_filters_by_base_model(client: TestClient, tmp_path: Path) -> None:
    models_dir = tmp_path / "models_lora"
    models_dir.mkdir(parents=True)
    lora_dir = models_dir / "author--style-lora"
    lora_dir.mkdir(parents=True)
    (lora_dir / "adapter_config.json").write_text(
        json.dumps({"base_model_name_or_path": "runwayml/stable-diffusion-v1-5"}),
        encoding="utf-8",
    )
    (lora_dir / "pytorch_lora_weights.safetensors").write_bytes(b"123")
    other_lora = models_dir / "author--other-lora"
    other_lora.mkdir(parents=True)
    (other_lora / "adapter_config.json").write_text(
        json.dumps({"base_model_name_or_path": "foo/bar"}),
        encoding="utf-8",
    )
    (other_lora / "pytorch_lora_weights.safetensors").write_bytes(b"456")

    settings = client.get("/api/settings").json()
    settings["paths"]["models_dir"] = str(models_dir.resolve())
    settings["defaults"]["text2image_model"] = "runwayml/stable-diffusion-v1-5"
    put_resp = client.put("/api/settings", json=settings)
    assert put_resp.status_code == 200

    resp = client.get("/api/models/loras/catalog", params={"task": "text-to-image", "model_ref": "runwayml/stable-diffusion-v1-5"})
    assert resp.status_code == 200
    ids = {item["id"] for item in resp.json()["items"]}
    assert "author/style-lora" in ids
    assert "author/other-lora" not in ids


def test_lora_catalog_filters_unknown_video_lora_by_lineage(client: TestClient, tmp_path: Path) -> None:
    models_dir = tmp_path / "models_video_lora_filter"
    models_dir.mkdir(parents=True)
    # No adapter_config.json, so this LoRA has no explicit base hint.
    lora_dir = models_dir / "author--lcm-lora-sdxl"
    lora_dir.mkdir(parents=True)
    (lora_dir / "pytorch_lora_weights.safetensors").write_bytes(b"123")

    settings = client.get("/api/settings").json()
    settings["paths"]["models_dir"] = str(models_dir.resolve())
    settings["defaults"]["text2video_model"] = "damo-vilab/text-to-video-ms-1.7b"
    put_resp = client.put("/api/settings", json=settings)
    assert put_resp.status_code == 200

    resp = client.get("/api/models/loras/catalog", params={"task": "text-to-video", "model_ref": "damo-vilab/text-to-video-ms-1.7b"})
    assert resp.status_code == 200
    ids = {item["id"] for item in resp.json()["items"]}
    assert "author/lcm-lora-sdxl" not in ids


def test_text2video_worker_skips_incompatible_lora(client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    outputs_dir = tmp_path / "outputs_t2v"
    models_dir = tmp_path / "models_t2v"
    tmp_dir = tmp_path / "tmp_t2v"
    settings = client.get("/api/settings").json()
    settings["paths"]["outputs_dir"] = str(outputs_dir.resolve())
    settings["paths"]["models_dir"] = str(models_dir.resolve())
    settings["paths"]["tmp_dir"] = str(tmp_dir.resolve())
    settings["defaults"]["text2video_model"] = "damo-vilab/text-to-video-ms-1.7b"
    put_resp = client.put("/api/settings", json=settings)
    assert put_resp.status_code == 200

    class FakeText2VideoPipeline:
        def load_lora_weights(self, *_args, **_kwargs):
            raise AttributeError("'UNet3DConditionModel' object has no attribute 'load_lora_adapter'")

        def __call__(self, **_kwargs):
            frames = [np.zeros((16, 16, 3), dtype=np.uint8) for _ in range(8)]
            return type("FakeOut", (), {"frames": [frames]})()

    def fake_export(_frames, output_path: Path, fps: int = 0, **_kwargs) -> str:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"\x00\x00\x00\x18ftypmp42")
        return "fake_codec"

    monkeypatch.setattr(main, "get_pipeline", lambda kind, model_ref, settings_payload: FakeText2VideoPipeline())
    monkeypatch.setattr(main, "get_device_and_dtype", lambda: ("cuda", "float16"))
    monkeypatch.setattr(main, "export_video_with_fallback", fake_export)

    task_id = main.create_task("text2video", "Generation queued")
    payload = main.Text2VideoRequest(
        prompt="test prompt",
        model_id="",
        lora_ids=["author/lcm-lora-sdxl"],
        num_inference_steps=2,
        num_frames=8,
        guidance_scale=9.0,
        fps=8,
        seed=None,
        backend="cuda",
    )

    main.text2video_worker(task_id, payload)
    last = client.get(f"/api/tasks/{task_id}").json()
    assert last["status"] == "completed"
    assert last["result"]["loras"] == []
    assert (outputs_dir / last["result"]["video_file"]).exists()


def test_recent_logs_endpoint(client: TestClient) -> None:
    resp = client.get("/api/logs/recent", params={"lines": 50})
    assert resp.status_code == 200
    body = resp.json()
    assert "log_file" in body
    assert Path(body["log_file"]).exists()
    assert isinstance(body.get("lines"), list)


def test_clear_hf_cache_endpoint(client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    hub = tmp_path / "hub"
    transformers = tmp_path / "transformers"
    hub.mkdir(parents=True)
    transformers.mkdir(parents=True)
    (hub / "x.bin").write_bytes(b"123")
    (transformers / "y.bin").write_bytes(b"456")

    monkeypatch.setattr(main, "gather_hf_cache_candidates", lambda: {hub, transformers})

    resp = client.post("/api/cache/hf/clear", json={"dry_run": False})
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert str(hub.resolve()) in body["removed_paths"]
    assert str(transformers.resolve()) in body["removed_paths"]
    assert not hub.exists()
    assert not transformers.exists()

===== FILE: tests/unit/test_helpers.py =====

import pytest
from pathlib import Path
import json

import main

pytestmark = pytest.mark.unit


def test_sanitize_desanitize_roundtrip() -> None:
    repo_id = "org-name/model-name"
    sanitized = main.sanitize_repo_id(repo_id)
    restored = main.desanitize_repo_id(sanitized)
    assert sanitized == "org-name--model-name"
    assert restored == repo_id


def test_deep_merge_nested() -> None:
    base = {"a": {"x": 1, "y": 2}, "b": 3}
    updates = {"a": {"y": 9}, "c": 4}
    merged = main.deep_merge(base, updates)
    assert merged == {"a": {"x": 1, "y": 9}, "b": 3, "c": 4}
    assert base == {"a": {"x": 1, "y": 2}, "b": 3}


def test_parse_civitai_model_id() -> None:
    assert main.parse_civitai_model_id("civitai/12345") == 12345
    assert main.parse_civitai_model_id("CivitAI/987") == 987
    assert main.parse_civitai_model_id("foo/bar") is None
    assert main.parse_civitai_model_id("civitai/not-number") is None


def test_find_local_preview_relpath(tmp_path: Path) -> None:
    models_dir = tmp_path / "models"
    model_dir = models_dir / "foo--bar"
    model_dir.mkdir(parents=True)
    preview = model_dir / "thumbnail.png"
    preview.write_bytes(b"fake")
    rel = main.find_local_preview_relpath(model_dir=model_dir, models_dir=models_dir)
    assert rel == "foo--bar/thumbnail.png"


def test_is_local_lora_dir_ignores_base_pipeline_with_lora_named_file(tmp_path: Path) -> None:
    model_dir = tmp_path / "base-model"
    model_dir.mkdir(parents=True)
    (model_dir / "model_index.json").write_text(
        json.dumps({"_class_name": "StableDiffusionXLPipeline"}),
        encoding="utf-8",
    )
    (model_dir / "sd_xl_offset_example-lora_1.0.safetensors").write_bytes(b"abc")
    assert main.is_local_lora_dir(model_dir) is False


def test_cache_get_or_set_uses_cached_value(monkeypatch: pytest.MonkeyPatch) -> None:
    main.SEARCH_API_CACHE.clear()
    counter = {"calls": 0}

    def loader() -> dict:
        counter["calls"] += 1
        return {"value": counter["calls"]}

    first = main.cache_get_or_set("test", {"a": 1}, loader)
    second = main.cache_get_or_set("test", {"a": 1}, loader)
    assert first["value"] == 1
    assert second["value"] == 1
    assert counter["calls"] == 1
