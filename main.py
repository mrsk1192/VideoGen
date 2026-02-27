import contextlib
import copy
import ctypes
import dataclasses
import gc
import hashlib
import importlib.util
import inspect
import json
import logging
import math
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
from urllib.request import Request as UrlRequest
from urllib.request import urlopen

import httpx
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from huggingface_hub import HfApi, hf_hub_download, snapshot_download
from pydantic import BaseModel, Field

from videogen.config import (
    DEFAULT_SETTINGS as VIDEOGEN_DEFAULT_SETTINGS,
)
from videogen.config import (
    SettingsStore as VideoGenSettingsStore,
)
from videogen.config import (
    deep_merge as config_deep_merge,
)
from videogen.config import (
    ensure_runtime_dirs as config_ensure_runtime_dirs,
)
from videogen.config import (
    parse_bool_setting as config_parse_bool_setting,
)
from videogen.config import (
    resolve_path as config_resolve_path,
)
from videogen.config import (
    sanitize_settings as config_sanitize_settings,
)
from videogen.runtime import (
    apply_pre_torch_env,
    select_device_and_dtype,
)
from videogen.runtime import (
    runtime_diagnostics as runtime_diagnostics_snapshot,
)
from videogen.storage import run_cleanup
from videogen.tasks import (
    TaskCancelledError,
    TaskManager,
)
from videogen.tasks import (
    task_progress_heartbeat as task_progress_heartbeat_ctx,
)

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
DATA_DIR = BASE_DIR / "data"
PRE_TORCH_ENV = apply_pre_torch_env(BASE_DIR)

TORCH_IMPORT_ERROR: Optional[str] = None
DIFFUSERS_IMPORT_ERROR: Optional[str] = None
torch: Any = None
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

DEFAULT_SETTINGS: Dict[str, Any] = copy.deepcopy(VIDEOGEN_DEFAULT_SETTINGS)

TASKS: Dict[str, Dict[str, Any]] = {}
TASKS_LOCK = threading.Lock()
TASK_MANAGER = TaskManager()
PIPELINES: Dict[str, Any] = {}
PIPELINES_LOCK = threading.Lock()
PIPELINE_LOAD_LOCK = threading.Lock()
PIPELINE_USAGE_COUNTS: Dict[str, int] = {}
GPU_SEMAPHORE_LOCK = threading.Lock()
GPU_GENERATION_SEMAPHORE = threading.Semaphore(1)
GPU_SEMAPHORE_LIMIT = 1
VAES: Dict[str, Any] = {}
VAES_LOCK = threading.Lock()
MODEL_SIZE_CACHE: Dict[str, Dict[str, Any]] = {}
MODEL_SIZE_CACHE_LOCK = threading.Lock()
MODEL_SIZE_CACHE_TTL_SEC = 60 * 60
SEARCH_API_CACHE: Dict[str, Dict[str, Any]] = {}
SEARCH_API_CACHE_LOCK = threading.Lock()
SEARCH_API_CACHE_TTL_SEC = 60 * 5
LOCAL_TREE_CACHE: Dict[str, Dict[str, Any]] = {}
LOCAL_TREE_CACHE_LOCK = threading.Lock()
LOCAL_TREE_CACHE_TTL_SEC = 30
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
    "/api/models/local/tree",
    "/api/models/local/rescan",
    "/api/models/local/reveal",
    "/api/tasks",
    "/api/tasks/cancel",
    "/api/runtime",
    "/api/cleanup",
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
    "text-to-video": {"TextToVideoSDPipeline", "WanPipeline"},
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
LOCAL_MODEL_FILE_EXTENSIONS = {".safetensors", ".ckpt", ".pt", ".pth", ".bin"}
LOCAL_TREE_TASK_ORDER = ["T2I", "I2I", "T2V", "V2V"]
LOCAL_TREE_TASK_TO_API = {
    "T2I": "text-to-image",
    "I2I": "image-to-image",
    "T2V": "text-to-video",
    "V2V": "image-to-video",
}
LOCAL_TREE_API_TO_TASK = {value: key for key, value in LOCAL_TREE_TASK_TO_API.items()}
LOCAL_TREE_TASK_ALIASES = {
    "T2I": "T2I",
    "TEXT2IMAGE": "T2I",
    "TEXT_TO_IMAGE": "T2I",
    "I2I": "I2I",
    "IMAGE2IMAGE": "I2I",
    "IMAGE_TO_IMAGE": "I2I",
    "T2V": "T2V",
    "TEXT2VIDEO": "T2V",
    "TEXT_TO_VIDEO": "T2V",
    "V2V": "V2V",
    "I2V": "V2V",
    "IMAGE2VIDEO": "V2V",
    "IMAGE_TO_VIDEO": "V2V",
}
LOCAL_TREE_CATEGORY_ORDER = ["BaseModel", "Lora", "VAE"]
VIDEO_DURATION_SECONDS_DEFAULT = 2.0
VIDEO_DURATION_SECONDS_MAX = 10800.0
FRAMEPACK_SEGMENT_FRAMES_DEFAULT = 16
FRAMEPACK_SEGMENT_FRAMES_MIN = 4
FRAMEPACK_SEGMENT_FRAMES_MAX = 128
FRAMEPACK_OVERLAP_FRAMES_DEFAULT = 2
FRAMEPACK_LONG_VIDEO_SECONDS_THRESHOLD = 30.0 * 60.0
FRAMEPACK_LONG_SEGMENT_FRAMES_DEFAULT = 8
# Backward-compatible aliases for existing environment variables and diagnostics.
VIDEO_CHUNK_FRAMES_DEFAULT = FRAMEPACK_SEGMENT_FRAMES_DEFAULT
VIDEO_CHUNK_FRAMES_MIN = FRAMEPACK_SEGMENT_FRAMES_MIN
VIDEO_CHUNK_FRAMES_MAX = FRAMEPACK_SEGMENT_FRAMES_MAX


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
    return config_deep_merge(base, updates)


def parse_bool_setting(raw_value: Any, default: bool = False) -> bool:
    return config_parse_bool_setting(raw_value, default=default)


def sanitize_settings(payload: Dict[str, Any]) -> Dict[str, Any]:
    return config_sanitize_settings(payload)


def resolve_path(path_like: str) -> Path:
    return config_resolve_path(path_like, BASE_DIR)


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
    return detect_runtime()


def host_memory_stats() -> Dict[str, int]:
    stats: Dict[str, int] = {}
    # Windows: use GlobalMemoryStatusEx for available physical memory.
    if os.name == "nt":
        with contextlib.suppress(Exception):

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
                stats["host_total_bytes"] = int(status.ullTotalPhys)
                stats["host_available_bytes"] = int(status.ullAvailPhys)
                return stats
    # POSIX fallback.
    with contextlib.suppress(Exception):
        page_size = int(os.sysconf("SC_PAGE_SIZE"))
        total_pages = int(os.sysconf("SC_PHYS_PAGES"))
        avail_pages = int(os.sysconf("SC_AVPHYS_PAGES"))
        if page_size > 0 and total_pages > 0:
            stats["host_total_bytes"] = page_size * total_pages
            stats["host_available_bytes"] = max(0, page_size * max(0, avail_pages))
    return stats


def memory_stats_snapshot() -> Dict[str, int]:
    stats = host_memory_stats()
    stats.update(gpu_memory_stats())
    return stats


def log_memory_snapshot(label: str, *, kind: str, source: str, strategy: str) -> None:
    stats = memory_stats_snapshot()
    if not stats:
        return
    LOGGER.info(
        "pipeline load memory kind=%s strategy=%s stage=%s source=%s host_avail=%s host_total=%s gpu_free=%s gpu_total=%s allocated=%s reserved=%s",
        kind,
        strategy,
        label,
        source,
        stats.get("host_available_bytes"),
        stats.get("host_total_bytes"),
        stats.get("gpu_free_bytes"),
        stats.get("gpu_total_bytes"),
        stats.get("torch_allocated_bytes"),
        stats.get("torch_reserved_bytes"),
    )


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


def iter_exception_chain(exc: BaseException) -> list[BaseException]:
    chain: list[BaseException] = []
    visited: set[int] = set()
    current: Optional[BaseException] = exc
    while current is not None:
        marker = id(current)
        if marker in visited:
            break
        visited.add(marker)
        chain.append(current)
        current = current.__cause__ or current.__context__
    return chain


def is_gpu_oom_error(exc: BaseException) -> bool:
    for current in iter_exception_chain(exc):
        if TORCH_IMPORT_ERROR is None:
            with contextlib.suppress(Exception):
                if isinstance(current, torch.OutOfMemoryError):
                    return True
        text = str(current).lower()
        if "out of memory" not in text:
            continue
        if ("cuda" in text) or ("hip" in text) or ("vram" in text) or ("hbm" in text):
            return True
    return False


def format_user_friendly_error(exc: BaseException) -> str:
    if is_gpu_oom_error(exc):
        return "GPU out of memory. Try reducing duration/frames/resolution/steps, " "or set lower gpu_max_concurrency in settings."
    return str(exc)


def detach_exception_state(exc: BaseException) -> None:
    with contextlib.suppress(Exception):
        tb = exc.__traceback__
        if tb is not None:
            traceback.clear_frames(tb)
    with contextlib.suppress(Exception):
        exc.__traceback__ = None
    with contextlib.suppress(Exception):
        exc.__cause__ = None
    with contextlib.suppress(Exception):
        exc.__context__ = None


def clear_cuda_allocator_cache(reason: str) -> None:
    if TORCH_IMPORT_ERROR or not torch.cuda.is_available():
        return
    with contextlib.suppress(Exception):
        torch.cuda.empty_cache()
    with contextlib.suppress(Exception):
        torch.cuda.ipc_collect()
    LOGGER.info("cuda allocator cache cleared reason=%s", reason)


def _increment_pipeline_usage_locked(cache_key: str) -> None:
    PIPELINE_USAGE_COUNTS[cache_key] = int(PIPELINE_USAGE_COUNTS.get(cache_key, 0)) + 1


def release_pipeline_usage(cache_key: Optional[str]) -> None:
    key = str(cache_key or "").strip()
    if not key:
        return
    with PIPELINES_LOCK:
        current = int(PIPELINE_USAGE_COUNTS.get(key, 0))
        if current <= 1:
            PIPELINE_USAGE_COUNTS.pop(key, None)
        else:
            PIPELINE_USAGE_COUNTS[key] = current - 1


def unload_unused_cached_components(
    *,
    keep_cache_keys: Optional[set[str]] = None,
    clear_vaes: bool = False,
    reason: str = "",
) -> Dict[str, int]:
    keep = keep_cache_keys or set()
    evicted_items: list[tuple[str, Any]] = []
    with PIPELINES_LOCK:
        for cache_key in list(PIPELINES.keys()):
            if cache_key in keep:
                continue
            if int(PIPELINE_USAGE_COUNTS.get(cache_key, 0)) > 0:
                continue
            pipe = PIPELINES.pop(cache_key)
            evicted_items.append((cache_key, pipe))
    for _, pipe in evicted_items:
        with contextlib.suppress(Exception):
            if hasattr(pipe, "to"):
                pipe.to("cpu")
        with contextlib.suppress(Exception):
            if getattr(pipe, "_videogen_lora_loaded", False) and hasattr(pipe, "unload_lora_weights"):
                pipe.unload_lora_weights()
                pipe._videogen_lora_loaded = False
    released_vaes = 0
    if clear_vaes:
        in_use_now = False
        with PIPELINES_LOCK:
            in_use_now = any(int(v) > 0 for v in PIPELINE_USAGE_COUNTS.values())
        if not in_use_now:
            vae_items: list[Any] = []
            with VAES_LOCK:
                for source in list(VAES.keys()):
                    vae_items.append(VAES.pop(source))
            released_vaes = len(vae_items)
            for vae in vae_items:
                with contextlib.suppress(Exception):
                    if hasattr(vae, "to"):
                        vae.to("cpu")
    with contextlib.suppress(Exception):
        gc.collect()
    clear_cuda_allocator_cache(reason=f"{reason}:post-evict")
    evicted_count = len(evicted_items)
    if evicted_count or released_vaes:
        LOGGER.info(
            "released cached components reason=%s pipelines=%s vaes=%s keep=%s",
            reason or "(none)",
            evicted_count,
            released_vaes,
            len(keep),
        )
    return {"pipelines": evicted_count, "vaes": released_vaes}


def cleanup_before_generation_load(
    *,
    kind: Literal["text-to-image", "image-to-image", "text-to-video", "image-to-video"],
    model_ref: str,
    settings: Dict[str, Any],
    clear_vaes: bool = True,
) -> Dict[str, int]:
    source = resolve_model_source(model_ref, settings)
    keep_cache_keys = {f"{kind}:{source}"}
    released = unload_unused_cached_components(
        keep_cache_keys=keep_cache_keys,
        clear_vaes=clear_vaes,
        reason=f"before-generate:{kind}",
    )
    LOGGER.info(
        "pre-generation cleanup kind=%s model_ref=%s source=%s released_pipelines=%s released_vaes=%s",
        kind,
        model_ref,
        source,
        released.get("pipelines"),
        released.get("vaes"),
    )
    return released


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


def is_deletable_local_model_path(target: Path) -> bool:
    if not target.exists():
        return False
    if target.is_file():
        return is_local_tree_model_file(target)
    if target.is_dir():
        return is_legacy_local_model_candidate(target) or is_deletable_model_dir(target)
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


settings_store = VideoGenSettingsStore(DATA_DIR / "settings.json", DEFAULT_SETTINGS)


def ensure_runtime_dirs(settings: Dict[str, Any]) -> None:
    config_ensure_runtime_dirs(settings, BASE_DIR)


def refresh_gpu_generation_semaphore(settings: Dict[str, Any]) -> None:
    global GPU_GENERATION_SEMAPHORE, GPU_SEMAPHORE_LIMIT
    server_settings = settings.get("server", {}) if isinstance(settings, dict) else {}
    try:
        requested = int(server_settings.get("gpu_max_concurrency", 1))
    except Exception:
        requested = 1
    requested = max(1, min(requested, 8))
    with GPU_SEMAPHORE_LOCK:
        if requested == GPU_SEMAPHORE_LIMIT:
            return
        GPU_GENERATION_SEMAPHORE = threading.Semaphore(requested)
        GPU_SEMAPHORE_LIMIT = requested
    LOGGER.info("gpu generation concurrency updated max_parallel=%s", requested)


ensure_runtime_dirs(settings_store.get())
setup_logger(settings_store.get())
refresh_gpu_generation_semaphore(settings_store.get())


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
                (p for p in providers if ("NPU" in p.upper()) or ("VITISAI" in p.upper()) or ("RYZENAI" in p.upper())),
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
    diffusers_error = DIFFUSERS_IMPORT_ERROR
    if TORCH_IMPORT_ERROR is None and diffusers_error is None:
        try:
            load_diffusers_components()
        except Exception as exc:  # pragma: no cover
            diffusers_error = str(exc)
    runtime_info = runtime_diagnostics_snapshot(
        settings=settings,
        torch_module=torch,
        import_error=TORCH_IMPORT_ERROR,
        diffusers_error=diffusers_error,
        npu_available=npu_available,
        npu_backend=npu_backend,
        npu_reason=npu_reason,
        t2v_backend_default=configured_t2v_backend,
        t2v_npu_runner_configured=npu_runner_configured,
    )
    runtime_info["pre_torch_env"] = PRE_TORCH_ENV
    runtime_info["gpu_max_concurrency_effective"] = GPU_SEMAPHORE_LIMIT
    expected_aotriton = "1" if parse_bool_setting(settings.get("server", {}).get("rocm_aotriton_experimental", True), True) else "0"
    current_aotriton = str(os.environ.get("TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL", "")).strip()
    if current_aotriton != expected_aotriton:
        runtime_info["aotriton_mismatch"] = {
            "expected": expected_aotriton,
            "actual": current_aotriton or "(unset)",
            "warning": "AOTriton env differs from settings. Restart with start.bat or set env before torch import.",
        }
    return runtime_info


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


def resolve_video_timing(
    *,
    fps: int,
    duration_seconds: Optional[float],
    legacy_num_frames: Optional[int],
) -> tuple[int, float]:
    fps_value = max(1, int(fps))
    legacy_frames = int(legacy_num_frames) if legacy_num_frames is not None else 0
    if legacy_frames > 0:
        total_frames = legacy_frames
        resolved_duration = float(total_frames) / float(fps_value)
        return total_frames, resolved_duration
    raw_duration = VIDEO_DURATION_SECONDS_DEFAULT if duration_seconds is None else float(duration_seconds)
    resolved_duration = max(0.1, min(VIDEO_DURATION_SECONDS_MAX, raw_duration))
    total_frames = max(1, int(resolved_duration * float(fps_value) + 0.5))
    resolved_duration = float(total_frames) / float(fps_value)
    return total_frames, resolved_duration


def resolve_framepack_plan(*, total_frames: int, fps: int) -> Dict[str, Any]:
    total = max(1, int(total_frames))
    fps_value = max(1, int(fps))
    long_video_threshold_frames = max(1, int(FRAMEPACK_LONG_VIDEO_SECONDS_THRESHOLD * float(fps_value)))
    long_video_mode = total >= long_video_threshold_frames

    raw_segment = os.environ.get("VIDEOGEN_FRAMEPACK_SEGMENT_FRAMES", "").strip()
    if not raw_segment:
        # Keep compatibility with previous naming.
        raw_segment = os.environ.get("VIDEOGEN_VIDEO_CHUNK_FRAMES", str(FRAMEPACK_SEGMENT_FRAMES_DEFAULT)).strip()
    try:
        segment_frames = int(raw_segment)
    except Exception:
        segment_frames = FRAMEPACK_SEGMENT_FRAMES_DEFAULT
    segment_frames = max(FRAMEPACK_SEGMENT_FRAMES_MIN, min(FRAMEPACK_SEGMENT_FRAMES_MAX, segment_frames))

    if long_video_mode:
        raw_long_segment = os.environ.get(
            "VIDEOGEN_FRAMEPACK_LONG_SEGMENT_FRAMES",
            str(FRAMEPACK_LONG_SEGMENT_FRAMES_DEFAULT),
        ).strip()
        try:
            long_segment_frames = int(raw_long_segment)
        except Exception:
            long_segment_frames = FRAMEPACK_LONG_SEGMENT_FRAMES_DEFAULT
        long_segment_frames = max(
            FRAMEPACK_SEGMENT_FRAMES_MIN,
            min(FRAMEPACK_SEGMENT_FRAMES_MAX, long_segment_frames),
        )
        segment_frames = min(segment_frames, long_segment_frames)

    segment_frames = min(segment_frames, total)

    raw_overlap = os.environ.get(
        "VIDEOGEN_FRAMEPACK_OVERLAP_FRAMES",
        str(FRAMEPACK_OVERLAP_FRAMES_DEFAULT),
    ).strip()
    try:
        overlap_frames = int(raw_overlap)
    except Exception:
        overlap_frames = FRAMEPACK_OVERLAP_FRAMES_DEFAULT
    overlap_frames = max(0, min(overlap_frames, max(0, segment_frames - 1)))

    usable_frames_per_pack = max(1, segment_frames - overlap_frames)
    if total <= segment_frames:
        pack_count = 1
    else:
        remaining_after_first = total - segment_frames
        pack_count = 1 + ((remaining_after_first + usable_frames_per_pack - 1) // usable_frames_per_pack)
    return {
        "segment_frames": segment_frames,
        "overlap_frames": overlap_frames,
        "usable_frames_per_pack": usable_frames_per_pack,
        "pack_count": pack_count,
        "long_video_mode": long_video_mode,
    }


def configured_video_chunk_frames(total_frames: int) -> int:
    # Legacy helper kept for compatibility; FramePack now drives video segmentation.
    plan = resolve_framepack_plan(total_frames=total_frames, fps=1)
    return int(plan["segment_frames"])


def iter_framepack_segments(total_frames: int, segment_frames: int, overlap_frames: int) -> list[Dict[str, int]]:
    total = max(1, int(total_frames))
    segment = max(1, int(segment_frames))
    overlap = max(0, min(int(overlap_frames), max(0, segment - 1)))
    segments: list[Dict[str, int]] = []
    produced = 0
    index = 0
    while produced < total:
        if index == 0:
            request_frames = min(segment, total)
            trim_head_frames = 0
        else:
            remaining = total - produced
            request_frames = min(segment, remaining + overlap)
            trim_head_frames = min(overlap, max(0, request_frames - 1))
        append_frames = max(1, request_frames - trim_head_frames)
        if append_frames > (total - produced):
            append_frames = total - produced
            trim_head_frames = max(0, request_frames - append_frames)
        segments.append(
            {
                "index": index + 1,
                "request_frames": request_frames,
                "trim_head_frames": trim_head_frames,
                "append_frames": append_frames,
                "produced_before": produced,
            }
        )
        produced += append_frames
        index += 1
    return segments


def run_text2video_npu_runner(task_id: str, payload: "Text2VideoRequest", settings: Dict[str, Any], model_ref: str) -> Dict[str, Any]:
    runner_raw = str(settings.get("server", {}).get("t2v_npu_runner", "")).strip()
    if not runner_raw:
        raise RuntimeError("NPU backend requires server.t2v_npu_runner. " "Configure a runner executable/script path in Settings.")
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
    target_frames, resolved_duration_seconds = resolve_video_timing(
        fps=int(payload.fps),
        duration_seconds=payload.duration_seconds,
        legacy_num_frames=payload.num_frames,
    )
    req_payload = {
        "task_id": task_id,
        "prompt": payload.prompt,
        "negative_prompt": payload.negative_prompt or "",
        "model_id": model_ref,
        "npu_model_dir": npu_model_dir,
        "num_inference_steps": int(payload.num_inference_steps),
        "duration_seconds": float(resolved_duration_seconds),
        "num_frames": int(target_frames),
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
        raise RuntimeError(f"NPU runner failed (exit={completed.returncode}). " f"stderr_tail={stderr_tail or '(none)'}")
    if not output_path.exists():
        raise RuntimeError(f"NPU runner completed but output video not found: {output_path}")
    return {"video_file": output_name, "encoder": "npu_runner", "runner": str(runner_path)}


def get_device_and_dtype() -> tuple[str, Any]:
    settings = settings_store.get()
    device, dtype, _ = select_device_and_dtype(
        settings=settings,
        torch_module=torch,
        import_error=TORCH_IMPORT_ERROR,
    )
    return device, dtype


@contextlib.contextmanager
def inference_execution_context(device: str, dtype: Any) -> Any:
    """
    推論実行コンテキスト。

    なぜ必要か:
    - ROCm/NVIDIA ともに `inference_mode` で不要な勾配追跡を止めてメモリ負荷を下げる。
    - GPU時は `autocast` を併用し、設定された mixed precision を確実に適用する。
    - CPU時は autocast を使わず、推論モードのみを適用する。
    """
    resolved_dtype = dtype
    if isinstance(dtype, str):
        mapped_dtype = {
            "float16": torch.float16,
            "fp16": torch.float16,
            "bf16": torch.bfloat16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
            "fp32": torch.float32,
        }.get(dtype.strip().lower())
        if mapped_dtype is not None:
            resolved_dtype = mapped_dtype
    with torch.inference_mode():
        if device == "cuda":
            with torch.autocast(device_type="cuda", dtype=resolved_dtype):
                yield
            return
        yield


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
            AutoencoderKL,
            AutoPipelineForImage2Image,
            AutoPipelineForText2Image,
            DiffusionPipeline,
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
            "DiffusionPipeline": DiffusionPipeline,
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
    task_id = TASK_MANAGER.create(task_type, message)
    LOGGER.info("task created id=%s type=%s message=%s", task_id, task_type, message)
    return task_id


def update_task(task_id: str, **updates: Any) -> None:
    previous = TASK_MANAGER.get(task_id)
    TASK_MANAGER.update(task_id, **updates)
    current = TASK_MANAGER.get(task_id)
    if not previous or not current:
        return
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
    with task_progress_heartbeat_ctx(
        manager=TASK_MANAGER,
        task_id=task_id,
        start_progress=start_progress,
        end_progress=end_progress,
        message=message,
        interval_sec=interval_sec,
        estimated_duration_sec=estimated_duration_sec,
    ):
        yield


def get_task(task_id: str) -> Optional[Dict[str, Any]]:
    return TASK_MANAGER.get(task_id)


def is_task_cancelled(task_id: str) -> bool:
    return TASK_MANAGER.is_cancel_requested(task_id)


def ensure_task_not_cancelled(task_id: str) -> None:
    TASK_MANAGER.check_cancelled(task_id)


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


def load_pipeline_from_pretrained_with_strategy(
    *,
    loader: Callable[..., Any],
    source: str,
    dtype: Any,
    prefer_gpu_device_map: bool,
    kind: str,
) -> Any:
    base_kwargs: Dict[str, Any] = {"torch_dtype": dtype}
    offload_dir_raw = os.environ.get("VIDEOGEN_PRETRAINED_OFFLOAD_DIR", "").strip()
    if offload_dir_raw:
        offload_dir = resolve_path(offload_dir_raw)
    else:
        offload_dir = resolve_path("tmp") / "pretrained_offload"
    with contextlib.suppress(Exception):
        offload_dir.mkdir(parents=True, exist_ok=True)

    gpu_first_base_kwargs: Dict[str, Any] = {
        **base_kwargs,
        "low_cpu_mem_usage": True,
        "offload_state_dict": True,
        "offload_folder": str(offload_dir),
    }
    attempts: list[tuple[str, Dict[str, Any]]] = []
    allow_fallback = parse_bool_setting(os.environ.get("VIDEOGEN_ALLOW_PRETRAINED_LOAD_FALLBACK", "0"), default=False)
    if prefer_gpu_device_map:
        attempts.append(
            (
                "device_map_cuda_safetensors",
                {
                    **gpu_first_base_kwargs,
                    "device_map": "cuda",
                    "use_safetensors": True,
                },
            )
        )
        attempts.append(
            (
                "device_map_cuda",
                {
                    **gpu_first_base_kwargs,
                    "device_map": "cuda",
                },
            )
        )
    if (not prefer_gpu_device_map) or allow_fallback:
        attempts.append(
            (
                "low_cpu_mem_usage",
                {
                    **base_kwargs,
                    "low_cpu_mem_usage": True,
                    "offload_state_dict": True,
                    "offload_folder": str(offload_dir),
                },
            )
        )
        attempts.append(("baseline", dict(base_kwargs)))

    last_error: Optional[Exception] = None
    for strategy_name, kwargs in attempts:
        with contextlib.suppress(Exception):
            gc.collect()
        clear_cuda_allocator_cache(reason=f"pretrained-load-before:{kind}:{strategy_name}")
        log_memory_snapshot("before", kind=kind, source=source, strategy=strategy_name)
        try:
            LOGGER.info(
                "pipeline pretrained load attempt kind=%s strategy=%s source=%s kwargs=%s",
                kind,
                strategy_name,
                source,
                ",".join(sorted(kwargs.keys())),
            )
            pipe = loader(source, **kwargs)
            LOGGER.info(
                "pipeline pretrained load success kind=%s strategy=%s source=%s",
                kind,
                strategy_name,
                source,
            )
            log_memory_snapshot("after-success", kind=kind, source=source, strategy=strategy_name)
            return pipe
        except TypeError as exc:
            # Some pipelines on older diffusers stacks reject strategy-specific kwargs.
            last_error = RuntimeError(f"{exc.__class__.__name__}: {exc}")
            LOGGER.warning(
                "pipeline pretrained load unsupported kwargs kind=%s strategy=%s source=%s error=%s",
                kind,
                strategy_name,
                source,
                str(exc),
            )
            detach_exception_state(exc)
            log_memory_snapshot("after-typeerror", kind=kind, source=source, strategy=strategy_name)
            continue
        except Exception as exc:
            last_error = RuntimeError(f"{exc.__class__.__name__}: {exc}")
            LOGGER.warning(
                "pipeline pretrained load failed kind=%s strategy=%s source=%s error=%s",
                kind,
                strategy_name,
                source,
                str(exc),
                exc_info=True,
            )
            if is_gpu_oom_error(exc):
                with contextlib.suppress(Exception):
                    gc.collect()
                clear_cuda_allocator_cache(reason=f"pretrained-load-oom:{kind}:{strategy_name}")
            detach_exception_state(exc)
            log_memory_snapshot("after-failure", kind=kind, source=source, strategy=strategy_name)
            continue
    if last_error is not None:
        if prefer_gpu_device_map and not allow_fallback:
            raise RuntimeError(
                "GPU-first pipeline loading failed and CPU-heavy fallback is disabled. "
                "Set VIDEOGEN_ALLOW_PRETRAINED_LOAD_FALLBACK=1 to allow fallback loading."
            ) from last_error
        raise last_error
    raise RuntimeError(f"failed to load pipeline kind={kind} source={source}")


def get_pipeline(
    kind: Literal["text-to-image", "image-to-image", "text-to-video", "image-to-video"],
    model_ref: str,
    settings: Dict[str, Any],
    *,
    acquire_usage: bool = False,
) -> Any:
    source = resolve_model_source(model_ref, settings)
    source_path = Path(source)
    source_is_single_file = is_single_file_model(source_path)
    cache_key = f"{kind}:{source}"
    with PIPELINES_LOCK:
        if cache_key in PIPELINES:
            LOGGER.info("pipeline cache hit kind=%s source=%s", kind, source)
            cached = PIPELINES[cache_key]
            with contextlib.suppress(Exception):
                cached._videogen_cache_key = cache_key
            if acquire_usage:
                _increment_pipeline_usage_locked(cache_key)
            return cached
    device, dtype = get_device_and_dtype()
    components = load_diffusers_components()
    AutoPipelineForText2Image = components["AutoPipelineForText2Image"]
    AutoPipelineForImage2Image = components["AutoPipelineForImage2Image"]
    StableDiffusionPipeline = components["StableDiffusionPipeline"]
    StableDiffusionImg2ImgPipeline = components["StableDiffusionImg2ImgPipeline"]
    StableDiffusionXLPipeline = components["StableDiffusionXLPipeline"]
    StableDiffusionXLImg2ImgPipeline = components["StableDiffusionXLImg2ImgPipeline"]
    TextToVideoSDPipeline = components["TextToVideoSDPipeline"]
    DiffusionPipeline = components["DiffusionPipeline"]
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
                with contextlib.suppress(Exception):
                    existing._videogen_cache_key = cache_key
                if acquire_usage:
                    _increment_pipeline_usage_locked(cache_key)
                return existing
        is_video_kind = kind in {"text-to-video", "image-to-video"}
        if is_video_kind and device == "cuda":
            unload_unused_cached_components(
                keep_cache_keys={cache_key},
                clear_vaes=True,
                reason=f"before-load:{kind}",
            )

        def _instantiate_pipeline() -> Any:
            if kind == "text-to-image":
                if source_is_single_file:
                    family = infer_single_file_family(source_path)
                    if family == "sdxl":
                        config = find_local_sdxl_base_config(settings)
                        kwargs: Dict[str, Any] = {"torch_dtype": dtype}
                        if config:
                            kwargs["config"] = config
                        return StableDiffusionXLPipeline.from_single_file(source, **kwargs)
                    return StableDiffusionPipeline.from_single_file(source, torch_dtype=dtype)
                return AutoPipelineForText2Image.from_pretrained(source, torch_dtype=dtype)
            if kind == "image-to-image":
                if source_is_single_file:
                    family = infer_single_file_family(source_path)
                    if family == "sdxl":
                        config = find_local_sdxl_base_config(settings)
                        kwargs = {"torch_dtype": dtype}
                        if config:
                            kwargs["config"] = config
                        return StableDiffusionXLImg2ImgPipeline.from_single_file(source, **kwargs)
                    return StableDiffusionImg2ImgPipeline.from_single_file(source, torch_dtype=dtype)
                return AutoPipelineForImage2Image.from_pretrained(source, torch_dtype=dtype)
            if kind == "text-to-video":
                loaded = load_pipeline_from_pretrained_with_strategy(
                    loader=lambda src, **kwargs: DiffusionPipeline.from_pretrained(src, **kwargs),
                    source=source,
                    dtype=dtype,
                    prefer_gpu_device_map=True,
                    kind=kind,
                )
                if isinstance(loaded, TextToVideoSDPipeline) and hasattr(loaded, "scheduler") and hasattr(loaded.scheduler, "config"):
                    loaded.scheduler = DPMSolverMultistepScheduler.from_config(loaded.scheduler.config)
                return loaded
            class_name = ""
            if source_path.exists() and source_path.is_dir():
                model_index = load_local_model_index(source_path)
                class_name = str((model_index or {}).get("_class_name") or "").strip()
            if class_name == "WanImageToVideoPipeline":
                return load_pipeline_from_pretrained_with_strategy(
                    loader=lambda src, **kwargs: WanImageToVideoPipeline.from_pretrained(src, **kwargs),
                    source=source,
                    dtype=dtype,
                    prefer_gpu_device_map=True,
                    kind=kind,
                )
            return load_pipeline_from_pretrained_with_strategy(
                loader=lambda src, **kwargs: I2VGenXLPipeline.from_pretrained(src, **kwargs),
                source=source,
                dtype=dtype,
                prefer_gpu_device_map=True,
                kind=kind,
            )

        pipe: Any
        try:
            pipe = _instantiate_pipeline()
        except Exception as exc:
            if is_video_kind and device == "cuda" and is_gpu_oom_error(exc):
                LOGGER.warning(
                    "pipeline load OOM; evicting unused cache and retrying once kind=%s source=%s",
                    kind,
                    source,
                    exc_info=True,
                )
                unload_unused_cached_components(
                    keep_cache_keys={cache_key},
                    clear_vaes=True,
                    reason=f"retry-after-oom:{kind}",
                )
                detach_exception_state(exc)
                pipe = _instantiate_pipeline()
            else:
                raise
        hf_device_map = getattr(pipe, "hf_device_map", None)
        if not hf_device_map:
            pipe = pipe.to(device)
        else:
            with contextlib.suppress(Exception):
                LOGGER.info(
                    "pipeline uses hf_device_map kind=%s source=%s entries=%s",
                    kind,
                    source,
                    len(hf_device_map) if isinstance(hf_device_map, dict) else 1,
                )
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
                # pipe.unet.to(memory_format=torch.contiguous_format)
            except Exception:
                LOGGER.debug("channels_last optimization skipped", exc_info=True)
        with PIPELINES_LOCK:
            with contextlib.suppress(Exception):
                pipe._videogen_cache_key = cache_key
            PIPELINES[cache_key] = pipe
            if acquire_usage:
                _increment_pipeline_usage_locked(cache_key)
    LOGGER.info("pipeline load done kind=%s source=%s elapsed_ms=%.1f", kind, source, (time.perf_counter() - load_started) * 1000)
    return pipe


def get_pipeline_for_inference(
    kind: Literal["text-to-image", "image-to-image", "text-to-video", "image-to-video"],
    model_ref: str,
    settings: Dict[str, Any],
) -> Any:
    try:
        return get_pipeline(kind, model_ref, settings, acquire_usage=True)
    except TypeError as exc:
        text = str(exc)
        if ("acquire_usage" not in text) or ("unexpected keyword argument" not in text):
            raise
        LOGGER.debug("get_pipeline compatibility fallback used kind=%s model_ref=%s", kind, model_ref)
        return get_pipeline(kind, model_ref, settings)


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


def normalize_frame_to_uint8_rgb(frame: Any) -> Any:
    import numpy as np

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


def frame_to_pil_image(frame: Any) -> Any:
    return Image.fromarray(normalize_frame_to_uint8_rgb(frame), mode="RGB")


def detect_framepack_context_arg(pipe: Any) -> str:
    signature = inspect.signature(pipe.__call__)
    accepted = set(signature.parameters.keys())
    for name in ("image", "init_image", "first_frame", "conditioning_image"):
        if name in accepted:
            return name
    accepts_var_kwargs = any(param.kind == inspect.Parameter.VAR_KEYWORD for param in signature.parameters.values())
    return "image" if accepts_var_kwargs else ""


def open_hardware_video_writer(output_path: Path, fps: int) -> tuple[Any, str, Callable[[Any], Any]]:
    if os.name != "nt":
        raise RuntimeError("Hardware video encoding is only supported on Windows in this build.")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import imageio.v2 as imageio  # type: ignore
    except Exception as exc:
        raise RuntimeError(f"Hardware video encoder runtime is unavailable: {exc}") from exc

    def normalize_frame_for_encoder(frame: Any) -> Any:
        return normalize_frame_to_uint8_rgb(frame)

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
            return writer, codec, normalize_frame_for_encoder
        except Exception:
            LOGGER.warning("hardware writer open failed codec=%s path=%s", codec, str(output_path), exc_info=True)
            continue
    raise RuntimeError("No AMD hardware video codec was available (tried: h264_amf, hevc_amf).")


def open_video_writer_with_policy(output_path: Path, fps: int) -> tuple[Any, str, Callable[[Any], Any]]:
    try:
        return open_hardware_video_writer(output_path, fps)
    except Exception:
        settings = settings_store.get()
        allow_software = parse_bool_setting(settings.get("server", {}).get("allow_software_video_fallback", False), default=False)
        if not allow_software:
            raise
        try:
            import imageio.v2 as imageio  # type: ignore

            output_path.parent.mkdir(parents=True, exist_ok=True)
            writer = imageio.get_writer(
                str(output_path),
                format="FFMPEG",
                mode="I",
                fps=int(fps),
                codec="libx264",
                macro_block_size=1,
                ffmpeg_log_level="error",
                ffmpeg_params=["-pix_fmt", "yuv420p"],
            )
            LOGGER.warning("hardware encoder unavailable. using software fallback codec=libx264 path=%s", str(output_path))
            return writer, "libx264", normalize_frame_to_uint8_rgb
        except Exception as software_exc:
            raise RuntimeError("Failed to open both hardware and software video encoders.") from software_exc


def append_frames_to_video_writer(
    writer: Any,
    normalize_frame: Callable[[Any], Any],
    frames: Any,
    skip_head_frames: int = 0,
) -> tuple[int, Optional[tuple[int, ...]], Optional[str]]:
    appended = 0
    sample_shape: Optional[tuple[int, ...]] = None
    sample_dtype: Optional[str] = None
    skip = max(0, int(skip_head_frames))
    for frame_index, frame in enumerate(frames):
        if frame_index < skip:
            continue
        norm = normalize_frame(frame)
        writer.append_data(norm)
        if appended == 0:
            sample_shape = tuple(int(v) for v in norm.shape)
            sample_dtype = str(norm.dtype)
        appended += 1
    return appended, sample_shape, sample_dtype


def export_video_with_fallback(frames: Any, output_path: Path, fps: int) -> str:
    """
    Prefer AMD AMF hardware encoding on Windows.
    Software fallback is opt-in via settings.server.allow_software_video_fallback.
    """
    writer: Any = None
    encoder_name = ""
    try:
        writer, encoder_name, normalize_frame = open_hardware_video_writer(output_path, fps=int(fps))
        frame_count, sample_shape, sample_dtype = append_frames_to_video_writer(writer, normalize_frame, frames)
        if frame_count <= 0:
            raise RuntimeError("no frames were generated to encode")
        LOGGER.info(
            "video encoded with hardware codec=%s path=%s frames=%s fps=%s sample_shape=%s sample_dtype=%s",
            encoder_name,
            str(output_path),
            frame_count,
            fps,
            sample_shape,
            sample_dtype,
        )
        return encoder_name
    except Exception as hardware_exc:
        if output_path.exists():
            output_path.unlink(missing_ok=True)
        settings = settings_store.get()
        allow_software = parse_bool_setting(settings.get("server", {}).get("allow_software_video_fallback", False), default=False)
        if not allow_software:
            raise RuntimeError(
                "Hardware video encoder is unavailable. "
                "Set settings.server.allow_software_video_fallback=true to opt in to software encoding."
            ) from hardware_exc
        try:
            import imageio.v2 as imageio  # type: ignore

            output_path.parent.mkdir(parents=True, exist_ok=True)
            writer = imageio.get_writer(
                str(output_path),
                format="FFMPEG",
                mode="I",
                fps=int(fps),
                codec="libx264",
                macro_block_size=1,
                ffmpeg_log_level="error",
                ffmpeg_params=["-pix_fmt", "yuv420p"],
            )
            frame_count, sample_shape, sample_dtype = append_frames_to_video_writer(writer, normalize_frame_to_uint8_rgb, frames)
            if frame_count <= 0:
                raise RuntimeError("no frames were generated to encode")
            LOGGER.warning(
                "video encoded with software fallback codec=libx264 path=%s frames=%s fps=%s sample_shape=%s sample_dtype=%s",
                str(output_path),
                frame_count,
                fps,
                sample_shape,
                sample_dtype,
            )
            return "libx264"
        except Exception as software_exc:
            if output_path.exists():
                output_path.unlink(missing_ok=True)
            raise RuntimeError("Hardware video encoder failed and software fallback also failed.") from software_exc
    finally:
        if writer is not None:
            with contextlib.suppress(Exception):
                writer.close()


def try_patch_wan_ftfy_dependency(pipe: Any) -> bool:
    """Patch diffusers Wan pipeline module-level `ftfy` binding when missing."""
    pipe_name = type(pipe).__name__
    if pipe_name not in {"WanPipeline", "WanImageToVideoPipeline"}:
        return False
    try:
        from diffusers.pipelines.wan import pipeline_wan as wan_module
    except Exception:
        return False
    if getattr(wan_module, "ftfy", None) is not None:
        return False
    try:
        import ftfy as ftfy_module  # type: ignore
    except Exception:
        return False
    wan_module.ftfy = ftfy_module
    LOGGER.info("patched diffusers Wan pipeline module with ftfy dependency")
    return True


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
        try:
            return pipe(**filtered)
        except NameError as exc:
            if "ftfy" not in str(exc):
                raise
            if try_patch_wan_ftfy_dependency(pipe):
                return pipe(**filtered)
            raise RuntimeError("Wan pipeline runtime dependency is missing: install `ftfy` in this environment.") from exc
    max_retry = 4
    current_kwargs = dict(filtered)
    for _ in range(max_retry):
        try:
            return pipe(**current_kwargs)
        except NameError as exc:
            if "ftfy" not in str(exc):
                raise
            if try_patch_wan_ftfy_dependency(pipe):
                return pipe(**current_kwargs)
            raise RuntimeError("Wan pipeline runtime dependency is missing: install `ftfy` in this environment.") from exc
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
        ensure_task_not_cancelled(task_id)
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
    if (
        "stable-diffusion-xl" in text
        or "sdxl" in text
        or "realvisxl" in text
        or "-xl" in text
        or "_xl" in text
        or re.search(r"\bxl\b", text)
    ):
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


def safe_non_negative_float(value: Any) -> Optional[float]:
    try:
        parsed = float(value)
    except Exception:
        return None
    if not math.isfinite(parsed) or parsed < 0.0:
        return None
    return parsed


def normalize_size_filter_bounds(size_min_mb: Optional[float], size_max_mb: Optional[float]) -> tuple[Optional[int], Optional[int]]:
    min_mb = safe_non_negative_float(size_min_mb)
    max_mb = safe_non_negative_float(size_max_mb)
    min_bytes = int(min_mb * 1024 * 1024) if min_mb is not None and min_mb > 0 else None
    max_bytes = int(max_mb * 1024 * 1024) if max_mb is not None and max_mb > 0 else None
    if min_bytes is not None and max_bytes is not None and max_bytes < min_bytes:
        raise ValueError("size_max_mb must be greater than or equal to size_min_mb")
    return min_bytes, max_bytes


def filter_models_by_size_bytes(items: list[Dict[str, Any]], min_bytes: Optional[int], max_bytes: Optional[int]) -> list[Dict[str, Any]]:
    if min_bytes is None and max_bytes is None:
        return items
    filtered: list[Dict[str, Any]] = []
    for item in items:
        size_bytes = safe_int(item.get("size_bytes"))
        if size_bytes is None:
            continue
        if min_bytes is not None and size_bytes < min_bytes:
            continue
        if max_bytes is not None and size_bytes > max_bytes:
            continue
        filtered.append(item)
    return filtered


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


def directory_size_bytes_limited(root: Path, max_entries: int = 2000) -> Optional[int]:
    if not root.exists():
        return 0
    total = 0
    seen = 0
    for entry in root.rglob("*"):
        if not entry.is_file():
            continue
        seen += 1
        if seen > max_entries:
            return None
        try:
            total += int(entry.stat().st_size)
        except OSError:
            continue
    return total


def normalize_local_tree_task_dir(name: str) -> Optional[str]:
    key = re.sub(r"[^A-Za-z0-9_]+", "", str(name or "").upper())
    return LOCAL_TREE_TASK_ALIASES.get(key)


def normalize_local_tree_category(name: str) -> Optional[str]:
    key = re.sub(r"[^A-Za-z0-9_]+", "", str(name or "").upper())
    if key in {"BASEMODEL", "BASE", "MODEL", "CHECKPOINT"}:
        return "BaseModel"
    if key in {"LORA", "LORAS"}:
        return "Lora"
    if key in {"VAE", "VEA"}:
        return "VAE"
    return None


def is_local_tree_model_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in LOCAL_MODEL_FILE_EXTENSIONS


def is_diffusers_model_directory(path: Path) -> bool:
    if not path.is_dir():
        return False
    if (path / "model_index.json").exists():
        return True
    if (path / "config.json").exists():
        return True
    return False


def is_local_tree_item_directory(path: Path, category: str) -> bool:
    if not path.is_dir():
        return False
    if category == "Lora":
        return is_local_lora_dir(path)
    if category == "VAE":
        return is_local_vae_dir(path) or is_diffusers_model_directory(path)
    if is_diffusers_model_directory(path):
        return True
    # Accept directories that store a single-file checkpoint directly under BaseModel.
    with contextlib.suppress(OSError):
        for child in path.iterdir():
            if is_local_tree_model_file(child):
                return True
    return False


def detect_model_provider_from_path(path: Path) -> Optional[str]:
    candidates: list[Path] = []
    if path.is_dir():
        candidates.extend([path / "model_meta.json", path / "civitai_model.json"])
    else:
        candidates.extend([path.parent / "model_meta.json", path.parent / "civitai_model.json"])
    for candidate in candidates:
        if not candidate.exists():
            continue
        try:
            payload = json.loads(candidate.read_text(encoding="utf-8"))
        except Exception:
            continue
        source = str(payload.get("source") or "").strip().lower()
        if source in ("huggingface", "civitai"):
            return source
    return None


def load_local_videogen_meta(path: Path) -> Dict[str, Any]:
    candidates: list[Path] = []
    if path.is_dir():
        candidates.append(path / "videogen_meta.json")
    else:
        candidates.append(path.parent / "videogen_meta.json")
    for candidate in candidates:
        if not candidate.exists() or not candidate.is_file():
            continue
        try:
            payload = json.loads(candidate.read_text(encoding="utf-8"))
        except Exception:
            continue
        if isinstance(payload, dict):
            return payload
    return {}


def normalize_local_tree_category_from_kind(kind: str, repo_hint: str = "") -> str:
    raw = str(kind or "").strip().lower()
    if raw in ("lora", "loras"):
        return "Lora"
    if raw in ("vae", "vea"):
        return "VAE"
    if raw in ("checkpoint", "basemodel", "base", "model"):
        return "BaseModel"
    hint = str(repo_hint or "").lower()
    if "lora" in hint:
        return "Lora"
    if "vae" in hint or "vea" in hint:
        return "VAE"
    return "BaseModel"


def legacy_item_category(path: Path) -> str:
    if path.is_dir():
        if is_local_lora_dir(path):
            return "Lora"
        if is_local_vae_dir(path):
            return "VAE"
        meta = load_local_videogen_meta(path)
        if meta:
            return normalize_local_tree_category_from_kind(str(meta.get("category") or meta.get("model_kind") or ""), path.name)
    return "BaseModel"


def legacy_item_task_dirs(path: Path) -> list[str]:
    def append_task_if_valid(values: list[str], task_value: str) -> None:
        canonical = LOCAL_TREE_API_TO_TASK.get(task_value)
        if canonical and canonical not in values:
            values.append(canonical)

    def infer_task_dirs_from_hint(text: str) -> list[str]:
        value = str(text or "").strip().lower()
        if not value:
            return []
        inferred: list[str] = []
        if any(token in value for token in ("i2v", "image2video", "image-to-video", "v2v")):
            inferred.append("V2V")
        if any(token in value for token in ("t2v", "text2video", "text-to-video", "texttovideo")):
            inferred.append("T2V")
        if any(token in value for token in ("i2i", "image2image", "image-to-image", "img2img")):
            inferred.append("I2I")
        if any(token in value for token in ("t2i", "text2image", "text-to-image", "txt2img")):
            inferred.append("T2I")
        return inferred

    meta = load_local_videogen_meta(path)
    task_values: list[str] = []
    if meta:
        task_value = str(meta.get("task") or "").strip().lower()
        if task_value:
            append_task_if_valid(task_values, task_value)
    if path.is_file():
        for task in single_file_model_compatible_tasks(path):
            append_task_if_valid(task_values, task)
        for inferred in infer_task_dirs_from_hint(path.name):
            if inferred not in task_values:
                task_values.append(inferred)
        return task_values
    if not task_values and path.is_dir():
        model_meta = detect_local_model_meta(path)
        for task in model_meta.compatible_tasks:
            append_task_if_valid(task_values, task)
    if not task_values and path.is_dir():
        model_files = [child for child in path.iterdir() if is_local_tree_model_file(child)]
        for child in model_files:
            for task in single_file_model_compatible_tasks(child):
                append_task_if_valid(task_values, task)
            for inferred in infer_task_dirs_from_hint(child.name):
                if inferred not in task_values:
                    task_values.append(inferred)
        if not task_values and model_files:
            for inferred in infer_task_dirs_from_hint(path.name):
                if inferred not in task_values:
                    task_values.append(inferred)
    if not task_values:
        repo_hint = str(meta.get("repo_id") or path.name if isinstance(meta, dict) else path.name)
        for inferred in infer_task_dirs_from_hint(repo_hint):
            if inferred not in task_values:
                task_values.append(inferred)
    if not task_values:
        category = legacy_item_category(path)
        if category == "VAE":
            task_values.extend(["T2I", "I2I"])
        elif category == "Lora":
            task_values.extend(["T2I", "I2I", "T2V", "V2V"])
        elif category == "BaseModel":
            task_values.extend(["T2I", "I2I"])
    return task_values


def is_legacy_local_model_candidate(path: Path) -> bool:
    if path.is_file():
        return is_local_tree_model_file(path)
    if not path.is_dir():
        return False
    if (path / "videogen_meta.json").exists():
        return True
    if is_local_lora_dir(path) or is_local_vae_dir(path) or is_diffusers_model_directory(path):
        return True
    with contextlib.suppress(OSError):
        for child in path.iterdir():
            if is_local_tree_model_file(child):
                return True
    return False


def model_item_preview_url(path: Path, model_root: Path) -> Optional[str]:
    if path.is_dir():
        preview_rel = find_local_preview_relpath(path, model_root)
        if preview_rel:
            return f"/api/models/preview?rel={quote(preview_rel, safe='/')}&base_dir={quote(str(model_root), safe='/:\\\\')}"
    else:
        stem = path.stem.lower()
        for suffix in (".png", ".jpg", ".jpeg", ".webp"):
            candidate = path.with_suffix(suffix)
            if candidate.exists() and candidate.is_file():
                rel = candidate.resolve().relative_to(model_root.resolve())
                return f"/api/models/preview?rel={quote(str(rel).replace('\\', '/'), safe='/')}&base_dir={quote(str(model_root), safe='/:\\\\')}"
            alt = path.parent / f"{stem}.preview{suffix}"
            if alt.exists() and alt.is_file():
                rel = alt.resolve().relative_to(model_root.resolve())
                return f"/api/models/preview?rel={quote(str(rel).replace('\\', '/'), safe='/')}&base_dir={quote(str(model_root), safe='/:\\\\')}"
    return None


def build_local_tree_item(
    item_path: Path,
    model_root: Path,
    task_dir: str,
    base_name: str,
    category: str,
) -> Dict[str, Any]:
    is_dir = item_path.is_dir()
    size_bytes: Optional[int] = None
    if is_dir:
        size_bytes = directory_size_bytes_limited(item_path)
    else:
        with contextlib.suppress(OSError):
            size_bytes = int(item_path.stat().st_size)
    provider = detect_model_provider_from_path(item_path)
    task_api = LOCAL_TREE_TASK_TO_API.get(task_dir, "")
    class_name = ""
    model_id = item_path.stem if item_path.is_file() else desanitize_repo_id(item_path.name)
    item_meta = load_local_videogen_meta(item_path)
    if is_dir:
        meta = detect_local_model_meta(item_path)
        class_name = meta.class_name
        if not provider:
            provider_value = str(item_meta.get("source") or "").strip().lower()
            if provider_value in ("huggingface", "civitai"):
                provider = provider_value
    else:
        class_name = "SingleFileModel" if item_path.suffix.lower() in SINGLE_FILE_MODEL_EXTENSIONS else "ModelFile"
    model_url = None
    if provider == "huggingface":
        model_url = f"https://huggingface.co/{quote(model_id, safe='/')}"
    elif provider == "civitai":
        civitai_id = parse_civitai_model_id(model_id)
        if civitai_id:
            model_url = f"https://civitai.com/models/{civitai_id}"
    return {
        "name": item_path.name,
        "display_name": item_path.stem if item_path.is_file() else item_path.name,
        "path": str(item_path.resolve()),
        "is_dir": is_dir,
        "size_bytes": size_bytes,
        "provider": provider,
        "base_name": base_name,
        "category": category,
        "task_dir": task_dir,
        "task_api": task_api,
        "apply_supported": bool(
            task_api and (category in ("BaseModel", "Lora") or (category == "VAE" and task_api in ("text-to-image", "image-to-image")))
        ),
        "model_id": model_id,
        "preview_url": model_item_preview_url(item_path, model_root),
        "model_url": model_url,
        "compatible_tasks": [task_api] if task_api and category == "BaseModel" else [],
        "is_lora": category == "Lora",
        "is_vae": category == "VAE",
        "class_name": class_name,
        "repo_hint": model_id,
        "can_delete": is_deletable_local_model_path(item_path),
    }


def build_local_model_tree(model_root: Path) -> Dict[str, Any]:
    tasks_out: list[Dict[str, Any]] = []
    flat_items: list[Dict[str, Any]] = []
    if not model_root.exists():
        return {
            "model_root": str(model_root),
            "generated_at": utc_now(),
            "tasks": tasks_out,
            "flat_items": flat_items,
        }
    task_dirs: list[Path] = [entry for entry in model_root.iterdir() if entry.is_dir()]
    normalized_task_dirs: list[tuple[str, Path]] = []
    for task_dir in task_dirs:
        canonical = normalize_local_tree_task_dir(task_dir.name)
        if not canonical:
            continue
        normalized_task_dirs.append((canonical, task_dir))
    normalized_task_dirs.sort(
        key=lambda pair: (LOCAL_TREE_TASK_ORDER.index(pair[0]) if pair[0] in LOCAL_TREE_TASK_ORDER else 999, pair[1].name.lower())
    )
    for canonical_task, task_path in normalized_task_dirs:
        bases_out: list[Dict[str, Any]] = []
        task_item_count = 0
        base_dirs = [entry for entry in task_path.iterdir() if entry.is_dir()]
        base_dirs.sort(key=lambda path: path.name.lower())
        for base_dir in base_dirs:
            categories_out: list[Dict[str, Any]] = []
            base_item_count = 0
            category_dirs = [entry for entry in base_dir.iterdir() if entry.is_dir()]
            grouped_categories: Dict[str, list[Path]] = {}
            for category_dir in category_dirs:
                normalized_category = normalize_local_tree_category(category_dir.name)
                if not normalized_category:
                    continue
                grouped_categories.setdefault(normalized_category, []).append(category_dir)
            normalized_category_names = sorted(
                grouped_categories.keys(),
                key=lambda name: (
                    LOCAL_TREE_CATEGORY_ORDER.index(name) if name in LOCAL_TREE_CATEGORY_ORDER else 999,
                    name.lower(),
                ),
            )
            for category_name in normalized_category_names:
                items: list[Dict[str, Any]] = []
                category_paths = sorted(grouped_categories.get(category_name, []), key=lambda path: path.name.lower())
                for category_path in category_paths:
                    children = sorted(category_path.iterdir(), key=lambda path: (path.is_file(), path.name.lower()))
                    for child in children:
                        if child.is_file() and not is_local_tree_model_file(child):
                            continue
                        if child.is_dir() and not is_local_tree_item_directory(child, category_name):
                            continue
                        item = build_local_tree_item(
                            item_path=child,
                            model_root=model_root,
                            task_dir=canonical_task,
                            base_name=base_dir.name,
                            category=category_name,
                        )
                        items.append(item)
                        flat_items.append(item)
                if not items:
                    continue
                item_count = len(items)
                base_item_count += item_count
                categories_out.append(
                    {
                        "category": category_name,
                        "path": str(category_paths[0].resolve()),
                        "source_paths": [str(path.resolve()) for path in category_paths],
                        "item_count": item_count,
                        "items": items,
                    }
                )
            if not categories_out:
                continue
            task_item_count += base_item_count
            bases_out.append(
                {
                    "base_name": base_dir.name,
                    "path": str(base_dir.resolve()),
                    "item_count": base_item_count,
                    "categories": categories_out,
                }
            )
        if not bases_out:
            continue
        tasks_out.append(
            {
                "task": canonical_task,
                "path": str(task_path.resolve()),
                "item_count": task_item_count,
                "bases": bases_out,
            }
        )
    # Backward compatibility:
    # Include legacy models placed directly under model_root (old download layout)
    # into a synthetic base branch under compatible task(s).
    task_lookup: Dict[str, Dict[str, Any]] = {str(task.get("task") or ""): task for task in tasks_out}

    def ensure_task_node(task_name: str) -> Dict[str, Any]:
        existing = task_lookup.get(task_name)
        if existing:
            return existing
        created = {
            "task": task_name,
            "path": str((model_root / task_name).resolve()),
            "item_count": 0,
            "bases": [],
        }
        task_lookup[task_name] = created
        tasks_out.append(created)
        return created

    def ensure_base_node(task_node: Dict[str, Any], base_name: str) -> Dict[str, Any]:
        bases = task_node.setdefault("bases", [])
        for base in bases:
            if str(base.get("base_name") or "") == base_name:
                return base
        created = {
            "base_name": base_name,
            "path": str(model_root.resolve()),
            "item_count": 0,
            "categories": [],
        }
        bases.append(created)
        return created

    def ensure_category_node(base_node: Dict[str, Any], category: str) -> Dict[str, Any]:
        categories = base_node.setdefault("categories", [])
        for entry in categories:
            if str(entry.get("category") or "") == category:
                return entry
        created = {
            "category": category,
            "path": str(model_root.resolve()),
            "source_paths": [str(model_root.resolve())],
            "item_count": 0,
            "items": [],
        }
        categories.append(created)
        categories.sort(
            key=lambda item: (
                LOCAL_TREE_CATEGORY_ORDER.index(str(item.get("category") or ""))
                if str(item.get("category") or "") in LOCAL_TREE_CATEGORY_ORDER
                else 999
            )
        )
        return created

    for child in sorted(model_root.iterdir(), key=lambda path: (path.is_file(), path.name.lower())):
        if normalize_local_tree_task_dir(child.name):
            continue
        if not is_legacy_local_model_candidate(child):
            continue
        category = legacy_item_category(child)
        task_dirs = legacy_item_task_dirs(child)
        if not task_dirs:
            continue
        for task_dir in task_dirs:
            item = build_local_tree_item(
                item_path=child,
                model_root=model_root,
                task_dir=task_dir,
                base_name="Imported",
                category=category,
            )
            item["imported"] = True
            task_node = ensure_task_node(task_dir)
            base_node = ensure_base_node(task_node, "Imported")
            category_node = ensure_category_node(base_node, category)
            category_items = category_node.setdefault("items", [])
            if any(str(existing.get("path") or "") == item["path"] for existing in category_items):
                continue
            category_items.append(item)
            category_items.sort(key=lambda entry: str(entry.get("name") or "").lower())
            category_node["item_count"] = len(category_items)
            base_node["item_count"] = int(base_node.get("item_count") or 0) + 1
            task_node["item_count"] = int(task_node.get("item_count") or 0) + 1
            flat_items.append(item)
    tasks_out.sort(
        key=lambda task: (
            LOCAL_TREE_TASK_ORDER.index(str(task.get("task") or "")) if str(task.get("task") or "") in LOCAL_TREE_TASK_ORDER else 999,
            str(task.get("task") or "").lower(),
        )
    )
    for task in tasks_out:
        bases = list(task.get("bases") or [])
        bases.sort(key=lambda base: (0 if str(base.get("base_name") or "") != "Imported" else 1, str(base.get("base_name") or "").lower()))
        task["bases"] = bases
    return {
        "model_root": str(model_root),
        "generated_at": utc_now(),
        "tasks": tasks_out,
        "flat_items": flat_items,
    }


def get_local_model_tree_cached(model_root: Path, force_rescan: bool = False) -> Dict[str, Any]:
    key = str(model_root.resolve()).lower()
    now = time.time()
    if not force_rescan:
        with LOCAL_TREE_CACHE_LOCK:
            cached = LOCAL_TREE_CACHE.get(key)
            if cached and (now - float(cached.get("ts") or 0.0)) < LOCAL_TREE_CACHE_TTL_SEC:
                value = cached.get("value")
                if isinstance(value, dict):
                    return copy.deepcopy(value)
    built = build_local_model_tree(model_root)
    with LOCAL_TREE_CACHE_LOCK:
        LOCAL_TREE_CACHE[key] = {"ts": now, "value": copy.deepcopy(built)}
    return built


def invalidate_local_tree_cache(model_root: Optional[Path] = None) -> None:
    with LOCAL_TREE_CACHE_LOCK:
        if model_root is None:
            LOCAL_TREE_CACHE.clear()
            return
        key = str(model_root.resolve()).lower()
        LOCAL_TREE_CACHE.pop(key, None)


def civitai_request_json(path: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    query = urlencode(params or {}, doseq=True)
    url = f"{CIVITAI_API_BASE}/{path.lstrip('/')}"
    if query:
        url = f"{url}?{query}"
    # Compatibility path for tests that monkeypatch urllib.urlopen.
    if getattr(urlopen, "__module__", "") != "urllib.request":
        req = UrlRequest(url, headers={"User-Agent": "ROCm-VideoGen/1.0"})
        with contextlib.closing(urlopen(req, timeout=20)) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
        return payload if isinstance(payload, dict) else {}
    settings = settings_store.get()
    server_settings = settings.get("server", {})
    timeout_sec = float(server_settings.get("request_timeout_sec", 20.0))
    retry_count = int(server_settings.get("request_retry_count", 2))
    retry_backoff_sec = float(server_settings.get("request_retry_backoff_sec", 1.0))
    timeout = httpx.Timeout(timeout_sec, connect=min(timeout_sec, 10.0))
    last_error: Optional[Exception] = None
    for attempt in range(max(0, retry_count) + 1):
        try:
            response = httpx.get(
                url,
                headers={"User-Agent": "ROCm-VideoGen/1.0"},
                timeout=timeout,
                follow_redirects=True,
            )
            response.raise_for_status()
            payload = response.json()
            return payload if isinstance(payload, dict) else {}
        except Exception as exc:
            last_error = exc
            if attempt >= retry_count:
                break
            time.sleep(max(0.1, retry_backoff_sec) * (2**attempt))
    raise RuntimeError(f"CivitAI request failed: {last_error}")


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


def search_civitai_models(
    task: Literal["text-to-image", "image-to-image", "text-to-video", "image-to-video"], query: str, limit: int
) -> list[Dict[str, Any]]:
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
                            "format": (
                                file_entry.get("metadata", {}).get("format") if isinstance(file_entry.get("metadata"), dict) else None
                            ),
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
    # Compatibility path for tests and custom monkeypatches that replace urllib.urlopen.
    if getattr(urlopen, "__module__", "") != "urllib.request":
        req = UrlRequest(url, headers={"User-Agent": "ROCm-VideoGen/1.0"})
        downloaded = 0
        last_reported = 0
        last_reported_ts = 0.0
        with contextlib.closing(urlopen(req, timeout=60)) as resp, destination.open("wb") as out_file:
            total = safe_int(getattr(resp, "headers", {}).get("Content-Length")) or safe_int(total_bytes_hint)
            if total and total > 0:
                update_task(task_id, progress=0.0, message=progress_message, downloaded_bytes=0, total_bytes=total)
            while True:
                ensure_task_not_cancelled(task_id)
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
        return downloaded

    settings = settings_store.get()
    server_settings = settings.get("server", {})
    timeout_sec = float(server_settings.get("request_timeout_sec", 60.0))
    retry_count = int(server_settings.get("request_retry_count", 2))
    retry_backoff_sec = float(server_settings.get("request_retry_backoff_sec", 1.0))
    timeout = httpx.Timeout(timeout_sec, connect=min(timeout_sec, 10.0))
    last_error: Optional[Exception] = None

    for attempt in range(max(0, retry_count) + 1):
        downloaded = 0
        last_reported = 0
        last_reported_ts = 0.0
        if destination.exists():
            destination.unlink(missing_ok=True)
        try:
            ensure_task_not_cancelled(task_id)
            with httpx.stream(
                "GET",
                url,
                headers={"User-Agent": "ROCm-VideoGen/1.0"},
                timeout=timeout,
                follow_redirects=True,
            ) as response:
                response.raise_for_status()
                total = safe_int(response.headers.get("Content-Length")) or safe_int(total_bytes_hint)
                if total and total > 0:
                    update_task(task_id, progress=0.0, message=progress_message, downloaded_bytes=0, total_bytes=total)
                with destination.open("wb") as out_file:
                    for chunk in response.iter_bytes(chunk_size=DOWNLOAD_STREAM_CHUNK_BYTES):
                        ensure_task_not_cancelled(task_id)
                        if not chunk:
                            continue
                        out_file.write(chunk)
                        downloaded += len(chunk)
                        now = time.time()
                        should_report = (downloaded - last_reported >= (2 * DOWNLOAD_STREAM_CHUNK_BYTES)) or (
                            (now - last_reported_ts) >= 1.0
                        )
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
                    update_task(
                        task_id,
                        progress=min(0.99, downloaded / total),
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
                return downloaded
        except TaskCancelledError:
            if destination.exists():
                destination.unlink(missing_ok=True)
            raise
        except Exception as exc:
            last_error = exc
            if attempt >= retry_count:
                break
            time.sleep(max(0.1, retry_backoff_sec) * (2**attempt))
    raise RuntimeError(f"download failed: {last_error}")


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
    duration_seconds: float = Field(default=VIDEO_DURATION_SECONDS_DEFAULT, gt=0.0, le=VIDEO_DURATION_SECONDS_MAX)
    num_frames: Optional[int] = Field(default=None, ge=1, le=216000)
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
    task: Optional[Literal["text-to-image", "image-to-image", "text-to-video", "image-to-video"]] = None
    base_model: Optional[str] = None
    model_kind: Optional[str] = None


class DeleteLocalModelRequest(BaseModel):
    model_name: Optional[str] = None
    path: Optional[str] = None
    base_dir: Optional[str] = None


class DeleteOutputRequest(BaseModel):
    file_name: str = Field(min_length=1)


class ClearHfCacheRequest(BaseModel):
    dry_run: bool = False


class CleanupRequest(BaseModel):
    include_cache: bool = True


class RescanLocalModelsRequest(BaseModel):
    dir: Optional[str] = None


class RevealLocalModelRequest(BaseModel):
    path: str = Field(min_length=1)
    base_dir: Optional[str] = None


class CancelTaskRequest(BaseModel):
    task_id: str = Field(min_length=1)


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
    pipeline_cache_key: Optional[str] = None
    try:
        ensure_task_not_cancelled(task_id)
        cleanup_before_generation_load(kind="text-to-image", model_ref=model_ref, settings=settings, clear_vaes=True)
        update_task(task_id, status="running", progress=0.05, message="Loading model")
        with GPU_GENERATION_SEMAPHORE:
            ensure_task_not_cancelled(task_id)
            pipe = get_pipeline_for_inference("text-to-image", model_ref, settings)
            pipeline_cache_key = str(getattr(pipe, "_videogen_cache_key", "") or "")
            device, dtype = get_device_and_dtype()
            update_task(task_id, progress=0.1, message="Loading model")
            apply_vae_to_pipeline(pipe, vae_ref, settings, device=device, dtype=dtype)
            update_task(task_id, progress=0.15, message="Applying LoRA")
            apply_loras_to_pipeline(pipe, lora_refs, payload.lora_scale, settings)
            ensure_task_not_cancelled(task_id)
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
            with inference_execution_context(device, dtype):
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
    except TaskCancelledError:
        LOGGER.info("text2image cancelled task_id=%s", task_id)
        update_task(task_id, status="cancelled", progress=1.0, message="Cancelled")
    except Exception as exc:
        trace = format_exception_trace()
        LOGGER.exception("text2image failed task_id=%s model=%s diagnostics=%s", task_id, model_ref, runtime_diagnostics())
        update_task(task_id, status="error", message="Generation failed", error=format_user_friendly_error(exc), error_trace=trace)
    finally:
        release_pipeline_usage(pipeline_cache_key)


def text2video_worker(task_id: str, payload: Text2VideoRequest) -> None:
    settings = settings_store.get()
    ensure_runtime_dirs(settings)
    model_ref = payload.model_id or settings["defaults"]["text2video_model"]
    lora_refs = collect_lora_refs(payload.lora_id, payload.lora_ids)
    effective_backend = resolve_text2video_backend(payload.backend, settings)
    total_frames, resolved_duration_seconds = resolve_video_timing(
        fps=int(payload.fps),
        duration_seconds=payload.duration_seconds,
        legacy_num_frames=payload.num_frames,
    )
    framepack_plan = resolve_framepack_plan(total_frames=total_frames, fps=int(payload.fps))
    framepack_segments = iter_framepack_segments(
        total_frames=total_frames,
        segment_frames=int(framepack_plan["segment_frames"]),
        overlap_frames=int(framepack_plan["overlap_frames"]),
    )
    pack_count = len(framepack_segments)
    LOGGER.info(
        "text2video start task_id=%s model=%s backend=%s requested_backend=%s loras=%s lora_scale=%s steps=%s duration_sec=%.3f frames=%s fps=%s framepack_segment=%s framepack_overlap=%s framepack_packs=%s framepack_mode=%s guidance=%s seed=%s",
        task_id,
        model_ref,
        effective_backend,
        payload.backend,
        ",".join(lora_refs) if lora_refs else "(none)",
        payload.lora_scale,
        payload.num_inference_steps,
        resolved_duration_seconds,
        total_frames,
        payload.fps,
        framepack_plan["segment_frames"],
        framepack_plan["overlap_frames"],
        pack_count,
        "long" if framepack_plan["long_video_mode"] else "standard",
        payload.guidance_scale,
        payload.seed,
    )
    pipeline_cache_key: Optional[str] = None
    semaphore_acquired = False
    try:
        ensure_task_not_cancelled(task_id)
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
                    "duration_seconds": resolved_duration_seconds,
                    "total_frames": total_frames,
                    "framepack_segment_frames": int(framepack_plan["segment_frames"]),
                    "framepack_overlap_frames": int(framepack_plan["overlap_frames"]),
                    "framepack_pack_count": pack_count,
                    "framepack_mode": "npu_runner",
                },
            )
            LOGGER.info(
                "text2video done task_id=%s backend=npu output=%s runner=%s",
                task_id,
                result["video_file"],
                result.get("runner"),
            )
            return

        GPU_GENERATION_SEMAPHORE.acquire()
        semaphore_acquired = True
        cleanup_before_generation_load(kind="text-to-video", model_ref=model_ref, settings=settings, clear_vaes=True)
        update_task(task_id, status="running", progress=0.05, message="Loading model")
        try:
            pipe = get_pipeline_for_inference("text-to-video", model_ref, settings)
        except Exception as load_error:
            if effective_backend == "cuda" and is_gpu_oom_error(load_error):
                raise RuntimeError(f"Failed to load selected text-to-video model on GPU due to out-of-memory: {model_ref}") from load_error
            raise
        pipeline_cache_key = str(getattr(pipe, "_videogen_cache_key", "") or "")
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
        update_task(task_id, progress=0.26, message=f"Preparing video stream (0/{total_frames} frames)")
        device, dtype = get_device_and_dtype()
        output_name = f"text2video_{task_id}.mp4"
        output_path = resolve_path(settings["paths"]["outputs_dir"]) / output_name
        writer: Any = None
        encoder_name = ""
        normalize_frame: Optional[Callable[[Any], Any]] = None
        total_written = 0
        sample_shape: Optional[tuple[int, ...]] = None
        sample_dtype: Optional[str] = None
        try:
            writer, encoder_name, normalize_frame = open_video_writer_with_policy(output_path, fps=int(payload.fps))
            gen_device = "cuda" if device == "cuda" else "cpu"
            framepack_context_arg = detect_framepack_context_arg(pipe)
            carry_image: Optional[Any] = None
            for segment in framepack_segments:
                ensure_task_not_cancelled(task_id)
                segment_index = int(segment["index"])
                request_frames = int(segment["request_frames"])
                trim_head_frames = int(segment["trim_head_frames"])
                chunk_start = 0.3 + (0.62 * ((segment_index - 1) / max(pack_count, 1)))
                chunk_end = 0.3 + (0.62 * (segment_index / max(pack_count, 1)))
                update_task(
                    task_id,
                    progress=chunk_start,
                    message=f"Generating framepack {segment_index}/{pack_count} ({total_written}/{total_frames} frames)",
                )
                generator = None
                if payload.seed is not None:
                    generator = torch.Generator(device=gen_device).manual_seed(int(payload.seed) + segment_index - 1)
                step_progress_kwargs = build_step_progress_kwargs(
                    pipe=pipe,
                    task_id=task_id,
                    num_inference_steps=payload.num_inference_steps,
                    start_progress=chunk_start,
                    end_progress=max(chunk_start, chunk_end - 0.02),
                    message=f"Generating framepack {segment_index}/{pack_count}",
                )
                request_payload: Dict[str, Any] = {
                    "prompt": payload.prompt,
                    "negative_prompt": payload.negative_prompt or None,
                    "num_inference_steps": payload.num_inference_steps,
                    "num_frames": request_frames,
                    "guidance_scale": payload.guidance_scale,
                    "generator": generator,
                    "cross_attention_kwargs": {"scale": payload.lora_scale} if len(lora_refs) == 1 else None,
                    **step_progress_kwargs,
                }
                if carry_image is not None and framepack_context_arg:
                    request_payload[framepack_context_arg] = carry_image
                LOGGER.info(
                    "text2video framepack start task_id=%s segment=%s/%s request_frames=%s trim_head=%s context_arg=%s callback_keys=%s",
                    task_id,
                    segment_index,
                    pack_count,
                    request_frames,
                    trim_head_frames,
                    framepack_context_arg or "(none)",
                    ",".join(sorted(step_progress_kwargs.keys())) if step_progress_kwargs else "(none)",
                )
                gen_started = time.perf_counter()
                with inference_execution_context(device, dtype):
                    out = call_with_supported_kwargs(
                        pipe,
                        request_payload,
                    )
                chunk_frames = out.frames[0]
                appended, chunk_shape, chunk_dtype = append_frames_to_video_writer(
                    writer,
                    normalize_frame,
                    chunk_frames,
                    skip_head_frames=trim_head_frames,
                )
                if appended <= 0:
                    raise RuntimeError(f"No frames generated for framepack {segment_index}/{pack_count}")
                if sample_shape is None:
                    sample_shape = chunk_shape
                    sample_dtype = chunk_dtype
                total_written += appended
                with contextlib.suppress(Exception):
                    if isinstance(chunk_frames, list) and chunk_frames:
                        carry_image = frame_to_pil_image(chunk_frames[-1])
                LOGGER.info(
                    "text2video framepack done task_id=%s segment=%s/%s appended=%s total_written=%s elapsed_ms=%.1f",
                    task_id,
                    segment_index,
                    pack_count,
                    appended,
                    total_written,
                    (time.perf_counter() - gen_started) * 1000,
                )
                update_task(
                    task_id,
                    progress=chunk_end,
                    message=f"Encoded framepack {segment_index}/{pack_count} ({total_written}/{total_frames} frames)",
                )
                del out
                with contextlib.suppress(Exception):
                    del chunk_frames
                with contextlib.suppress(Exception):
                    del request_payload
                with contextlib.suppress(Exception):
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
        finally:
            if writer is not None:
                with contextlib.suppress(Exception):
                    writer.close()
        if total_written <= 0:
            raise RuntimeError("No frames were generated to encode")
        if total_written != total_frames:
            LOGGER.warning(
                "text2video frame count mismatch task_id=%s expected=%s actual=%s",
                task_id,
                total_frames,
                total_written,
            )
        LOGGER.info(
            "video encoded with hardware codec=%s path=%s frames=%s fps=%s sample_shape=%s sample_dtype=%s",
            encoder_name,
            str(output_path),
            total_written,
            payload.fps,
            sample_shape,
            sample_dtype,
        )
        update_task(
            task_id,
            status="completed",
            progress=1.0,
            message="Done",
            result={
                "video_file": output_name,
                "model": model_ref,
                "loras": lora_refs,
                "encoder": encoder_name,
                "backend": "cuda",
                "duration_seconds": resolved_duration_seconds,
                "total_frames": total_frames,
                "encoded_frames": total_written,
                "chunk_frames": int(framepack_plan["segment_frames"]),
                "framepack_segment_frames": int(framepack_plan["segment_frames"]),
                "framepack_overlap_frames": int(framepack_plan["overlap_frames"]),
                "framepack_pack_count": pack_count,
                "framepack_mode": "long" if framepack_plan["long_video_mode"] else "standard",
            },
        )
        LOGGER.info("text2video done task_id=%s backend=cuda output=%s", task_id, str(output_path))
    except TaskCancelledError:
        LOGGER.info("text2video cancelled task_id=%s", task_id)
        update_task(task_id, status="cancelled", progress=1.0, message="Cancelled")
    except Exception as exc:
        if "output_path" in locals() and isinstance(output_path, Path) and output_path.exists():
            with contextlib.suppress(Exception):
                output_path.unlink(missing_ok=True)
        trace = format_exception_trace()
        LOGGER.exception(
            "text2video failed task_id=%s model=%s backend=%s diagnostics=%s",
            task_id,
            model_ref,
            effective_backend,
            runtime_diagnostics(),
        )
        update_task(task_id, status="error", message="Generation failed", error=format_user_friendly_error(exc), error_trace=trace)
    finally:
        if semaphore_acquired:
            GPU_GENERATION_SEMAPHORE.release()
        release_pipeline_usage(pipeline_cache_key)


def image2video_worker(task_id: str, payload: Dict[str, Any]) -> None:
    settings = settings_store.get()
    ensure_runtime_dirs(settings)
    model_ref = payload["model_id"] or settings["defaults"]["image2video_model"]
    lora_refs = collect_lora_refs(payload.get("lora_id"), payload.get("lora_ids") or [])
    lora_scale = float(payload.get("lora_scale") or 1.0)
    image_path = Path(payload["image_path"])
    total_frames, resolved_duration_seconds = resolve_video_timing(
        fps=int(payload.get("fps") or 8),
        duration_seconds=float(payload.get("duration_seconds") or VIDEO_DURATION_SECONDS_DEFAULT),
        legacy_num_frames=int(payload.get("num_frames")) if payload.get("num_frames") not in (None, "") else None,
    )
    framepack_plan = resolve_framepack_plan(total_frames=total_frames, fps=int(payload.get("fps") or 8))
    framepack_segments = iter_framepack_segments(
        total_frames=total_frames,
        segment_frames=int(framepack_plan["segment_frames"]),
        overlap_frames=int(framepack_plan["overlap_frames"]),
    )
    pack_count = len(framepack_segments)
    LOGGER.info(
        "image2video start task_id=%s model=%s loras=%s lora_scale=%s image=%s steps=%s duration_sec=%.3f frames=%s fps=%s framepack_segment=%s framepack_overlap=%s framepack_packs=%s framepack_mode=%s guidance=%s size=%sx%s seed=%s",
        task_id,
        model_ref,
        ",".join(lora_refs) if lora_refs else "(none)",
        lora_scale,
        str(image_path),
        payload.get("num_inference_steps"),
        resolved_duration_seconds,
        total_frames,
        payload.get("fps"),
        framepack_plan["segment_frames"],
        framepack_plan["overlap_frames"],
        pack_count,
        "long" if framepack_plan["long_video_mode"] else "standard",
        payload.get("guidance_scale"),
        payload.get("width"),
        payload.get("height"),
        payload.get("seed"),
    )
    pipeline_cache_key: Optional[str] = None
    semaphore_acquired = False
    try:
        ensure_task_not_cancelled(task_id)
        GPU_GENERATION_SEMAPHORE.acquire()
        semaphore_acquired = True
        cleanup_before_generation_load(kind="image-to-video", model_ref=model_ref, settings=settings, clear_vaes=True)
        update_task(task_id, status="running", progress=0.05, message="Loading model")
        pipe = get_pipeline_for_inference("image-to-video", model_ref, settings)
        pipeline_cache_key = str(getattr(pipe, "_videogen_cache_key", "") or "")
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
        device, dtype = get_device_and_dtype()
        gen_device = "cuda" if device == "cuda" else "cpu"
        step_count = int(payload["num_inference_steps"])
        update_task(task_id, progress=0.3, message=f"Preparing video stream (0/{total_frames} frames)")
        output_name = f"image2video_{task_id}.mp4"
        output_path = resolve_path(settings["paths"]["outputs_dir"]) / output_name
        writer: Any = None
        encoder_name = ""
        normalize_frame: Optional[Callable[[Any], Any]] = None
        total_written = 0
        sample_shape: Optional[tuple[int, ...]] = None
        sample_dtype: Optional[str] = None
        try:
            writer, encoder_name, normalize_frame = open_video_writer_with_policy(output_path, fps=int(payload["fps"]))
            current_image = image
            for segment in framepack_segments:
                ensure_task_not_cancelled(task_id)
                segment_index = int(segment["index"])
                request_frames = int(segment["request_frames"])
                trim_head_frames = int(segment["trim_head_frames"])
                chunk_start = 0.34 + (0.58 * ((segment_index - 1) / max(pack_count, 1)))
                chunk_end = 0.34 + (0.58 * (segment_index / max(pack_count, 1)))
                update_task(
                    task_id,
                    progress=chunk_start,
                    message=f"Generating framepack {segment_index}/{pack_count} ({total_written}/{total_frames} frames)",
                )
                chunk_generator = None
                if payload["seed"] is not None:
                    chunk_generator = torch.Generator(device=gen_device).manual_seed(int(payload["seed"]) + segment_index - 1)
                step_progress_kwargs = build_step_progress_kwargs(
                    pipe=pipe,
                    task_id=task_id,
                    num_inference_steps=step_count,
                    start_progress=chunk_start,
                    end_progress=max(chunk_start, chunk_end - 0.02),
                    message=f"Generating framepack {segment_index}/{pack_count}",
                )
                LOGGER.info(
                    "image2video framepack start task_id=%s segment=%s/%s request_frames=%s trim_head=%s callback_keys=%s",
                    task_id,
                    segment_index,
                    pack_count,
                    request_frames,
                    trim_head_frames,
                    ",".join(sorted(step_progress_kwargs.keys())) if step_progress_kwargs else "(none)",
                )
                gen_started = time.perf_counter()
                with inference_execution_context(device, dtype):
                    out = call_with_supported_kwargs(
                        pipe,
                        {
                            "prompt": payload["prompt"],
                            "negative_prompt": payload["negative_prompt"] or None,
                            "image": current_image,
                            "height": height,
                            "width": width,
                            "target_fps": int(payload["fps"]),
                            "num_inference_steps": step_count,
                            "num_frames": request_frames,
                            "guidance_scale": float(payload["guidance_scale"]),
                            "generator": chunk_generator,
                            "cross_attention_kwargs": {"scale": lora_scale} if len(lora_refs) == 1 else None,
                            **step_progress_kwargs,
                        },
                    )
                chunk_frames = out.frames[0]
                appended, chunk_shape, chunk_dtype = append_frames_to_video_writer(
                    writer,
                    normalize_frame,
                    chunk_frames,
                    skip_head_frames=trim_head_frames,
                )
                if appended <= 0:
                    raise RuntimeError(f"No frames generated for framepack {segment_index}/{pack_count}")
                if sample_shape is None:
                    sample_shape = chunk_shape
                    sample_dtype = chunk_dtype
                total_written += appended
                with contextlib.suppress(Exception):
                    if isinstance(chunk_frames, list) and chunk_frames:
                        current_image = frame_to_pil_image(chunk_frames[-1])
                LOGGER.info(
                    "image2video framepack done task_id=%s segment=%s/%s appended=%s total_written=%s elapsed_ms=%.1f",
                    task_id,
                    segment_index,
                    pack_count,
                    appended,
                    total_written,
                    (time.perf_counter() - gen_started) * 1000,
                )
                update_task(
                    task_id,
                    progress=chunk_end,
                    message=f"Encoded framepack {segment_index}/{pack_count} ({total_written}/{total_frames} frames)",
                )
                del out
                with contextlib.suppress(Exception):
                    del chunk_frames
                with contextlib.suppress(Exception):
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
        finally:
            if writer is not None:
                with contextlib.suppress(Exception):
                    writer.close()
        if total_written <= 0:
            raise RuntimeError("No frames were generated to encode")
        if total_written != total_frames:
            LOGGER.warning(
                "image2video frame count mismatch task_id=%s expected=%s actual=%s",
                task_id,
                total_frames,
                total_written,
            )
        LOGGER.info(
            "video encoded with hardware codec=%s path=%s frames=%s fps=%s sample_shape=%s sample_dtype=%s",
            encoder_name,
            str(output_path),
            total_written,
            payload["fps"],
            sample_shape,
            sample_dtype,
        )
        update_task(
            task_id,
            status="completed",
            progress=1.0,
            message="Done",
            result={
                "video_file": output_name,
                "model": model_ref,
                "loras": lora_refs,
                "encoder": encoder_name,
                "duration_seconds": resolved_duration_seconds,
                "total_frames": total_frames,
                "encoded_frames": total_written,
                "chunk_frames": int(framepack_plan["segment_frames"]),
                "framepack_segment_frames": int(framepack_plan["segment_frames"]),
                "framepack_overlap_frames": int(framepack_plan["overlap_frames"]),
                "framepack_pack_count": pack_count,
                "framepack_mode": "long" if framepack_plan["long_video_mode"] else "standard",
            },
        )
        LOGGER.info("image2video done task_id=%s output=%s", task_id, str(output_path))
    except TaskCancelledError:
        LOGGER.info("image2video cancelled task_id=%s", task_id)
        update_task(task_id, status="cancelled", progress=1.0, message="Cancelled")
    except Exception as exc:
        if "output_path" in locals() and isinstance(output_path, Path) and output_path.exists():
            with contextlib.suppress(Exception):
                output_path.unlink(missing_ok=True)
        trace = format_exception_trace()
        LOGGER.exception("image2video failed task_id=%s model=%s diagnostics=%s", task_id, model_ref, runtime_diagnostics())
        update_task(task_id, status="error", message="Generation failed", error=format_user_friendly_error(exc), error_trace=trace)
    finally:
        if semaphore_acquired:
            GPU_GENERATION_SEMAPHORE.release()
        release_pipeline_usage(pipeline_cache_key)
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
    pipeline_cache_key: Optional[str] = None
    semaphore_acquired = False
    try:
        ensure_task_not_cancelled(task_id)
        GPU_GENERATION_SEMAPHORE.acquire()
        semaphore_acquired = True
        cleanup_before_generation_load(kind="image-to-image", model_ref=model_ref, settings=settings, clear_vaes=True)
        update_task(task_id, status="running", progress=0.05, message="Loading model")
        pipe = get_pipeline_for_inference("image-to-image", model_ref, settings)
        pipeline_cache_key = str(getattr(pipe, "_videogen_cache_key", "") or "")
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
        with inference_execution_context(device, dtype):
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
    except TaskCancelledError:
        LOGGER.info("image2image cancelled task_id=%s", task_id)
        update_task(task_id, status="cancelled", progress=1.0, message="Cancelled")
    except Exception as exc:
        trace = format_exception_trace()
        LOGGER.exception("image2image failed task_id=%s model=%s diagnostics=%s", task_id, model_ref, runtime_diagnostics())
        update_task(task_id, status="error", message="Generation failed", error=format_user_friendly_error(exc), error_trace=trace)
    finally:
        if semaphore_acquired:
            GPU_GENERATION_SEMAPHORE.release()
        release_pipeline_usage(pipeline_cache_key)
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
    if getattr(urlopen, "__module__", "") != "urllib.request":
        req = UrlRequest(preview_url, headers={"User-Agent": "ROCm-VideoGen/1.0"})
        with contextlib.closing(urlopen(req, timeout=20)) as resp, destination.open("wb") as out_file:
            while True:
                chunk = resp.read(1024 * 256)
                if not chunk:
                    break
                out_file.write(chunk)
        return
    settings = settings_store.get()
    timeout_sec = float(settings.get("server", {}).get("request_timeout_sec", 20.0))
    response = httpx.get(
        preview_url,
        headers={"User-Agent": "ROCm-VideoGen/1.0"},
        timeout=httpx.Timeout(timeout_sec, connect=min(timeout_sec, 10.0)),
        follow_redirects=True,
    )
    response.raise_for_status()
    destination.write_bytes(response.content)


def normalize_download_model_kind(value: str) -> str:
    normalized = str(value or "").strip().lower()
    if normalized in ("lora", "loras"):
        return "Lora"
    if normalized in ("vae", "vea"):
        return "VAE"
    return "BaseModel"


def infer_download_model_kind(repo_id: str, requested_kind: str, fallback_kind: str = "") -> str:
    if requested_kind:
        return normalize_download_model_kind(requested_kind)
    if fallback_kind:
        return normalize_download_model_kind(fallback_kind)
    repo_hint = str(repo_id or "").lower()
    if "lora" in repo_hint:
        return "Lora"
    if "vae" in repo_hint or "vea" in repo_hint:
        return "VAE"
    return "BaseModel"


def infer_download_base_model(repo_id: str, requested_base_model: str, fallback_hint: str = "") -> str:
    raw = str(requested_base_model or "").strip()
    if raw:
        return raw
    if fallback_hint:
        return fallback_hint
    estimated = infer_base_model_label(repo_id)
    return estimated if estimated != "Other" else "Unknown"


def write_videogen_model_meta(
    model_dir: Path,
    *,
    source: str,
    repo_id: str,
    task: str,
    base_model: str,
    category: str,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    payload: Dict[str, Any] = {
        "source": source,
        "repo_id": repo_id,
        "task": task,
        "base_model": base_model,
        "category": category,
        "downloaded_at": utc_now(),
    }
    if extra:
        payload.update({key: value for key, value in extra.items() if value is not None})
    (model_dir / "videogen_meta.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


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
    requested_task = str(req.task or "").strip().lower()
    if requested_task not in LOCAL_TREE_API_TO_TASK:
        requested_task = "text-to-image"
    requested_base_model = str(req.base_model or "").strip()
    requested_model_kind = str(req.model_kind or "").strip()
    LOGGER.info("download start task_id=%s repo=%s revision=%s target=%s", task_id, repo_id, revision or "main", str(model_dir))
    try:
        ensure_task_not_cancelled(task_id)
        update_task(task_id, status="running", progress=0.0, message=f"Preparing download {repo_id}", downloaded_bytes=0, total_bytes=None)

        explicit_source = str(req.source or "").strip().lower()
        civitai_model_id = req.civitai_model_id or parse_civitai_model_id(repo_id)
        is_civitai = explicit_source == "civitai" or civitai_model_id is not None
        if civitai_model_id is not None:
            ensure_task_not_cancelled(task_id)
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
            selected_version_base_model = str(selected_version.get("baseModel") or "").strip()
            write_videogen_model_meta(
                model_dir,
                source="civitai",
                repo_id=repo_id,
                task=requested_task,
                base_model=infer_download_base_model(repo_id, requested_base_model, selected_version_base_model),
                category=infer_download_model_kind(repo_id, requested_model_kind, str(selected_file.get("type") or "")),
                extra={
                    "model_id": civitai_model_id,
                    "version_id": selected_version.get("id"),
                    "file_id": selected_file.get("id"),
                },
            )
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
                if is_task_cancelled(task_id):
                    LOGGER.warning("HF snapshot download cancellation requested but cannot interrupt once started task_id=%s", task_id)
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
            write_videogen_model_meta(
                model_dir,
                source="huggingface",
                repo_id=repo_id,
                task=requested_task,
                base_model=infer_download_base_model(repo_id, requested_base_model),
                category=infer_download_model_kind(repo_id, requested_model_kind),
                extra={"revision": revision},
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
        invalidate_local_tree_cache(models_dir)
        LOGGER.info("download done task_id=%s repo=%s local_path=%s", task_id, repo_id, str(model_dir.resolve()))
    except TaskCancelledError:
        LOGGER.info("download cancelled task_id=%s repo=%s", task_id, repo_id)
        update_task(task_id, status="cancelled", progress=1.0, message="Cancelled")
    except Exception as exc:
        trace = format_exception_trace()
        LOGGER.exception("download failed task_id=%s repo=%s target=%s", task_id, repo_id, str(model_dir))
        update_task(task_id, status="error", message="Download failed", error=format_user_friendly_error(exc), error_trace=trace)


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
        LOGGER.warning("startup preload skipped on ROCm for stability. " "Set VIDEOGEN_ALLOW_ROCM_STARTUP_PRELOAD=1 to force.")
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


@app.get("/api/runtime")
def runtime_info() -> Dict[str, Any]:
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


@app.post("/api/cleanup")
def cleanup_runtime_storage(req: CleanupRequest) -> Dict[str, Any]:
    settings = settings_store.get()
    storage_settings = settings.get("storage", {})
    if not parse_bool_setting(storage_settings.get("cleanup_enabled", True), default=True):
        raise HTTPException(status_code=400, detail="cleanup is disabled by settings.storage.cleanup_enabled=false")
    cache_candidates = sorted(gather_hf_cache_candidates(), key=lambda path: str(path).lower()) if req.include_cache else []
    result = run_cleanup(settings=settings, base_dir=BASE_DIR, hf_cache_candidates=cache_candidates)
    LOGGER.info(
        "runtime cleanup done removed_outputs=%s removed_tmp=%s removed_cache_paths=%s",
        len(result.get("removed_outputs") or []),
        len(result.get("removed_tmp") or []),
        len(result.get("removed_cache_paths") or []),
    )
    return result


@app.put("/api/settings")
def update_settings(payload: Dict[str, Any]) -> Dict[str, Any]:
    updated = settings_store.update(payload)
    ensure_runtime_dirs(updated)
    setup_logger(updated)
    refresh_gpu_generation_semaphore(updated)
    desired_aotriton = "1" if parse_bool_setting(updated.get("server", {}).get("rocm_aotriton_experimental", True), True) else "0"
    current_aotriton = str(os.environ.get("TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL", "")).strip()
    if current_aotriton != desired_aotriton:
        os.environ["TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL"] = desired_aotriton
        LOGGER.warning(
            "updated TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL in current process to match settings (%s). "
            "Some runtimes still require process restart for full effect.",
            desired_aotriton,
        )
    LOGGER.info(
        "settings updated models_dir=%s outputs_dir=%s tmp_dir=%s logs_dir=%s log_level=%s listen_port=%s rocm_aotriton_experimental=%s require_gpu=%s allow_cpu_fallback=%s preload_default_t2i_on_startup=%s t2v_backend=%s t2v_npu_runner=%s t2v_npu_model_dir=%s preferred_dtype=%s gpu_max_concurrency=%s allow_software_video_fallback=%s",
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
        updated.get("server", {}).get("preferred_dtype"),
        updated.get("server", {}).get("gpu_max_concurrency"),
        updated.get("server", {}).get("allow_software_video_fallback"),
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
                        f"/api/models/preview?rel={quote(preview_rel, safe='/')}" f"&base_dir={quote(str(models_dir), safe='/:\\\\')}"
                    )
                items.append(
                    {
                        "name": child.name,
                        "repo_hint": desanitize_repo_id(child.name),
                        "path": str(child.resolve()),
                        "can_delete": is_deletable_local_model_path(child),
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
                        "can_delete": is_deletable_local_model_path(child),
                        "preview_url": None,
                        "class_name": "SingleFileCheckpoint",
                        "base_model": single_file_base_model_label(child),
                        "compatible_tasks": compatible_tasks,
                        "is_lora": False,
                        "is_vae": False,
                    }
                )
    return {"items": items, "base_dir": str(models_dir)}


@app.get("/api/models/local/tree")
def list_local_models_tree(dir: str = "") -> Dict[str, Any]:
    settings = settings_store.get()
    model_root = resolve_path(dir.strip() or settings["paths"]["models_dir"])
    tree = get_local_model_tree_cached(model_root, force_rescan=False)
    return tree


@app.post("/api/models/local/rescan")
def rescan_local_models(req: RescanLocalModelsRequest) -> Dict[str, Any]:
    settings = settings_store.get()
    model_root = resolve_path((req.dir or "").strip() or settings["paths"]["models_dir"])
    invalidate_local_tree_cache(model_root)
    tree = get_local_model_tree_cached(model_root, force_rescan=True)
    return tree


@app.post("/api/models/local/reveal")
def reveal_local_model(req: RevealLocalModelRequest) -> Dict[str, Any]:
    settings = settings_store.get()
    base_dir = resolve_path((req.base_dir or "").strip() or settings["paths"]["models_dir"])
    requested = Path(req.path).expanduser()
    if not requested.is_absolute():
        requested = (base_dir / requested).resolve()
    else:
        requested = requested.resolve()
    if not safe_in_directory(requested, base_dir):
        raise HTTPException(status_code=400, detail="Invalid target path")
    if not requested.exists():
        raise HTTPException(status_code=404, detail="Target not found")
    if os.name != "nt":
        return {"status": "not_supported", "reason": "reveal is only supported on Windows"}
    try:
        if requested.is_file():
            subprocess.Popen(["explorer", "/select,", str(requested)], shell=False)
        else:
            subprocess.Popen(["explorer", str(requested)], shell=False)
    except Exception as exc:
        LOGGER.warning("reveal local model failed path=%s", str(requested), exc_info=True)
        raise HTTPException(status_code=500, detail=f"Reveal failed: {exc}") from exc
    return {"status": "ok", "path": str(requested)}


@app.post("/api/models/local/delete")
def delete_local_model(req: DeleteLocalModelRequest) -> Dict[str, Any]:
    settings = settings_store.get()
    base_dir = resolve_path((req.base_dir or "").strip() or settings["paths"]["models_dir"])
    raw_path = str(req.path or "").strip()
    model_name = str(req.model_name or "").strip()
    if not raw_path and not model_name:
        raise HTTPException(status_code=400, detail="path or model_name is required")

    if raw_path:
        requested = Path(raw_path).expanduser()
        if not requested.is_absolute():
            target = (base_dir / requested).resolve()
        else:
            target = requested.resolve()
    else:
        if Path(model_name).name != model_name or "/" in model_name or "\\" in model_name:
            raise HTTPException(status_code=400, detail="Invalid model_name")
        target = (base_dir / model_name).resolve()
        if target.parent.resolve() != base_dir.resolve():
            raise HTTPException(status_code=400, detail="Only direct child directories can be deleted by model_name")

    if not safe_in_directory(target, base_dir):
        raise HTTPException(status_code=400, detail="Invalid target path")
    if target.resolve() == base_dir.resolve():
        raise HTTPException(status_code=400, detail="Base directory cannot be deleted")
    if not target.exists():
        raise HTTPException(status_code=404, detail="Target not found")
    if not is_deletable_local_model_path(target):
        raise HTTPException(status_code=400, detail="Target is not recognized as a deletable local model")

    if target.is_dir():
        shutil.rmtree(target)
    else:
        target.unlink(missing_ok=True)
    invalidate_local_tree_cache(base_dir)
    deleted_name = model_name or target.name
    LOGGER.info("local model deleted base_dir=%s name=%s path=%s", str(base_dir), deleted_name, str(target))
    return {"status": "ok", "deleted_path": str(target), "model_name": deleted_name}


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


def local_tree_flat_items_for_task(models_dir: Path, task: str) -> list[Dict[str, Any]]:
    tree = get_local_model_tree_cached(models_dir)
    task_dir = LOCAL_TREE_API_TO_TASK.get(task, "")
    flat_items = list(tree.get("flat_items") or [])
    filtered = [item for item in flat_items if str(item.get("task_dir") or "") == task_dir]
    filtered.sort(
        key=lambda item: (
            str(item.get("base_name") or "").lower(),
            str(item.get("category") or "").lower(),
            str(item.get("name") or "").lower(),
        )
    )
    return filtered


def build_catalog_item_from_tree(item: Dict[str, Any], label_prefix: str = "local") -> Dict[str, Any]:
    path_value = str(item.get("path") or "")
    path_obj = Path(path_value) if path_value else Path(".")
    repo_hint = str(item.get("repo_hint") or "").strip()
    if not repo_hint:
        repo_hint = path_obj.stem if path_obj.suffix else path_obj.name
    model_url = str(item.get("model_url") or "").strip() or None
    preview_url = str(item.get("preview_url") or "").strip() or None
    size_bytes = safe_int(item.get("size_bytes"))
    return {
        "source": "local",
        "label": f"[{label_prefix}] {repo_hint}",
        "value": path_value,
        "id": repo_hint,
        "size_bytes": size_bytes,
        "preview_url": preview_url,
        "model_url": model_url,
    }


@app.get("/api/models/catalog")
def model_catalog(task: Literal["text-to-image", "image-to-image", "text-to-video", "image-to-video"], limit: int = 30) -> Dict[str, Any]:
    settings = settings_store.get()
    models_dir = resolve_path(settings["paths"]["models_dir"])
    models_dir.mkdir(parents=True, exist_ok=True)
    capped_limit = min(max(limit, 1), 1000)
    tree = get_local_model_tree_cached(models_dir)
    flat_items = list(tree.get("flat_items") or [])
    task_dir = LOCAL_TREE_API_TO_TASK.get(task, "")
    ordered_items = [item for item in flat_items if str(item.get("task_dir") or "") == task_dir]
    ordered_items.sort(
        key=lambda item: (
            str(item.get("base_name") or "").lower(),
            str(item.get("name") or "").lower(),
        )
    )

    local_items: list[Dict[str, Any]] = []
    seen_paths: set[str] = set()
    for item in ordered_items:
        if str(item.get("category") or "") != "BaseModel":
            continue
        value = str(item.get("path") or "")
        if not value or value in seen_paths:
            continue
        seen_paths.add(value)
        local_items.append(build_catalog_item_from_tree(item, label_prefix="local"))
        if len(local_items) >= capped_limit:
            break
    default_model = get_default_model_for_task(task, settings)
    return {"items": local_items, "default_model": default_model}


@app.get("/api/models/search")
def search_models(
    task: Literal["text-to-image", "image-to-image", "text-to-video", "image-to-video"],
    query: str = "",
    limit: int = 30,
    source: Literal["all", "huggingface", "civitai"] = "all",
    base_model: str = "",
    size_min_mb: Optional[float] = None,
    size_max_mb: Optional[float] = None,
) -> Dict[str, Any]:
    settings = settings_store.get()
    token = settings["huggingface"].get("token") or None
    capped_limit = min(max(limit, 1), 50)
    try:
        min_size_bytes, max_size_bytes = normalize_size_filter_bounds(size_min_mb, size_max_mb)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
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
    results = filter_models_by_size_bytes(results, min_size_bytes, max_size_bytes)
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
    size_min_mb: Optional[float] = None,
    size_max_mb: Optional[float] = None,
) -> Dict[str, Any]:
    settings = settings_store.get()
    token = settings["huggingface"].get("token") or None
    capped_limit = min(max(limit, 1), 100)
    try:
        min_size_bytes, max_size_bytes = normalize_size_filter_bounds(size_min_mb, size_max_mb)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
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
    results = filter_models_by_size_bytes(results, min_size_bytes, max_size_bytes)

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
            "size_min_mb": size_min_mb,
            "size_max_mb": size_max_mb,
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
    task_dir = LOCAL_TREE_API_TO_TASK.get(task, "")
    tree = get_local_model_tree_cached(models_dir)
    flat_items = list(tree.get("flat_items") or [])
    task_items = [item for item in flat_items if str(item.get("category") or "") == "Lora" and str(item.get("task_dir") or "") == task_dir]
    other_items = [item for item in flat_items if str(item.get("category") or "") == "Lora" and str(item.get("task_dir") or "") != task_dir]
    ordered_items = task_items + other_items
    ordered_items.sort(key=lambda item: str(item.get("name") or "").lower())

    items: list[Dict[str, Any]] = []
    seen_paths: set[str] = set()
    for item in ordered_items:
        path_value = str(item.get("path") or "")
        if not path_value or path_value in seen_paths:
            continue
        child = Path(path_value)
        if not child.exists():
            continue
        seen_paths.add(path_value)
        base_model_hint = ""
        if child.is_dir():
            adapter_config = parse_lora_adapter_config(child)
            base_model_hint = str(adapter_config.get("base_model_name_or_path") or "").strip()
        if not base_model_hint:
            item_meta = load_local_videogen_meta(child)
            base_model_hint = str(item_meta.get("base_model") or "").strip()
        should_filter_by_model = task in ("text-to-image", "image-to-image")
        if should_filter_by_model and base_model_hint and not lora_matches_model(base_model_hint, effective_model):
            hint_lineage = infer_base_model_label(base_model_hint)
            model_lineage = infer_base_model_label(effective_model)
            if not (hint_lineage != "Other" and hint_lineage == model_lineage):
                continue
        repo_hint = str(item.get("repo_hint") or "").strip() or desanitize_repo_id(child.name)
        preview_url = str(item.get("preview_url") or "").strip() or None
        model_url = str(item.get("model_url") or "").strip() or None
        size_bytes = safe_int(item.get("size_bytes"))
        if size_bytes is None and child.is_dir():
            size_bytes = local_lora_size_bytes(child)
        items.append(
            {
                "source": "local",
                "label": f"[lora] {repo_hint}",
                "value": str(child.resolve()),
                "id": repo_hint,
                "base_model": base_model_hint or None,
                "size_bytes": size_bytes,
                "preview_url": preview_url,
                "model_url": model_url,
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
    tree = get_local_model_tree_cached(models_dir)
    flat_items = list(tree.get("flat_items") or [])
    flat_items.sort(key=lambda item: str(item.get("name") or "").lower())
    items: list[Dict[str, Any]] = []
    seen_paths: set[str] = set()
    for item in flat_items:
        if str(item.get("category") or "") != "VAE":
            continue
        path_value = str(item.get("path") or "")
        if not path_value or path_value in seen_paths:
            continue
        child = Path(path_value)
        if not child.exists():
            continue
        seen_paths.add(path_value)
        repo_hint = str(item.get("repo_hint") or "").strip() or desanitize_repo_id(child.name)
        preview_url = str(item.get("preview_url") or "").strip() or None
        model_url = str(item.get("model_url") or "").strip() or None
        items.append(
            {
                "source": "local",
                "label": f"[vae] {repo_hint}",
                "value": str(child.resolve()),
                "id": repo_hint,
                "size_bytes": safe_int(item.get("size_bytes")),
                "preview_url": preview_url,
                "model_url": model_url,
            }
        )
        if len(items) >= capped_limit:
            break
    return {"items": items}


@app.post("/api/models/download")
def download_model(req: DownloadRequest) -> Dict[str, str]:
    task_id = create_task("download", "Download queued")
    LOGGER.info(
        "download requested task_id=%s repo=%s source=%s revision=%s hf_revision=%s civitai_model_id=%s civitai_version_id=%s civitai_file_id=%s target_dir=%s task=%s base_model=%s model_kind=%s",
        task_id,
        req.repo_id,
        req.source or "",
        req.revision or "",
        req.hf_revision or "",
        req.civitai_model_id,
        req.civitai_version_id,
        req.civitai_file_id,
        req.target_dir,
        req.task or "",
        req.base_model or "",
        req.model_kind or "",
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
                detail=("NPU backend requires server.t2v_npu_runner. " "Set it in Settings (T2V NPU Runner)."),
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
        "text2video requested task_id=%s model=%s backend=%s requested_backend=%s loras=%s lora_scale=%s prompt_len=%s negative_len=%s steps=%s duration_sec=%.3f legacy_frames=%s guidance=%s fps=%s seed=%s",
        task_id,
        req.model_id or "(default)",
        backend,
        req.backend,
        ",".join(lora_refs) if lora_refs else "(none)",
        req.lora_scale,
        len(req.prompt or ""),
        len(req.negative_prompt or ""),
        req.num_inference_steps,
        float(req.duration_seconds),
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
    num_frames: Optional[int] = Form(None),
    duration_seconds: float = Form(VIDEO_DURATION_SECONDS_DEFAULT),
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
        "duration_seconds": duration_seconds,
        "guidance_scale": guidance_scale,
        "fps": fps,
        "width": width,
        "height": height,
        "seed": seed,
    }
    task_id = create_task("image2video", "Generation queued")
    LOGGER.info(
        "image2video requested task_id=%s model=%s loras=%s lora_scale=%s file=%s steps=%s duration_sec=%.3f legacy_frames=%s guidance=%s fps=%s size=%sx%s seed=%s",
        task_id,
        model_id.strip() or "(default)",
        ",".join(lora_refs) if lora_refs else "(none)",
        lora_scale,
        image.filename,
        num_inference_steps,
        float(duration_seconds),
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


@app.get("/api/tasks")
def list_tasks(
    task_type: str = "",
    status: str = "all",
    limit: int = 30,
) -> Dict[str, Any]:
    normalized_status = str(status or "all").strip().lower()
    allowed_status = {"all", "queued", "running", "completed", "error", "cancelled"}
    if normalized_status not in allowed_status:
        raise HTTPException(status_code=400, detail="Invalid status")
    return {"tasks": TASK_MANAGER.list(task_type=task_type, status=status, limit=limit)}


@app.post("/api/tasks/cancel")
def cancel_task(req: CancelTaskRequest) -> Dict[str, Any]:
    task_id = str(req.task_id or "").strip()
    if not task_id:
        raise HTTPException(status_code=400, detail="task_id is required")
    try:
        task = TASK_MANAGER.request_cancel(task_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Task not found") from exc
    return {"status": "ok", "task": task}


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
