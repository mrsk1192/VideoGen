import copy
import json
import threading
from pathlib import Path
from typing import Any, Dict, Optional

VIDEO_DURATION_SECONDS_DEFAULT = 2.0
VIDEO_DURATION_SECONDS_MAX = 10800.0
VALID_LOG_LEVELS = {"DEBUG", "INFO", "WARNING", "ERROR"}
SUPPORTED_T2V_BACKENDS = {"auto", "cuda", "npu"}
TRUE_BOOL_STRINGS = {"1", "true", "yes", "on"}
FALSE_BOOL_STRINGS = {"0", "false", "no", "off"}

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
        "gpu_max_concurrency": 1,
        "preferred_dtype": "float16",
        "allow_software_video_fallback": False,
        "request_timeout_sec": 20,
        "request_retry_count": 2,
        "request_retry_backoff_sec": 1.0,
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
        "duration_seconds": 2.0,
        "guidance_scale": 9.0,
        "fps": 8,
        "width": 512,
        "height": 512,
    },
}


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
    defaults = cleaned.setdefault("defaults", {})
    logging_config = cleaned.setdefault("logging", {})

    raw_port = server.get("listen_port", 8000)
    try:
        listen_port = int(raw_port)
    except Exception:
        listen_port = 8000
    server["listen_port"] = listen_port if 1 <= listen_port <= 65535 else 8000
    server["listen_host"] = str(server.get("listen_host", "0.0.0.0")).strip() or "0.0.0.0"
    server["rocm_aotriton_experimental"] = parse_bool_setting(
        server.get("rocm_aotriton_experimental", True), default=True
    )
    server["require_gpu"] = parse_bool_setting(server.get("require_gpu", True), default=True)
    server["allow_cpu_fallback"] = parse_bool_setting(server.get("allow_cpu_fallback", False), default=False)
    server["preload_default_t2i_on_startup"] = parse_bool_setting(
        server.get("preload_default_t2i_on_startup", True), default=True
    )
    server["allow_software_video_fallback"] = parse_bool_setting(
        server.get("allow_software_video_fallback", False), default=False
    )
    raw_t2v_backend = str(server.get("t2v_backend", "auto")).strip().lower()
    server["t2v_backend"] = raw_t2v_backend if raw_t2v_backend in SUPPORTED_T2V_BACKENDS else "auto"
    server["t2v_npu_runner"] = str(server.get("t2v_npu_runner", "")).strip()
    server["t2v_npu_model_dir"] = str(server.get("t2v_npu_model_dir", "")).strip()

    try:
        gpu_max_concurrency = int(server.get("gpu_max_concurrency", 1))
    except Exception:
        gpu_max_concurrency = 1
    server["gpu_max_concurrency"] = max(1, min(gpu_max_concurrency, 8))

    preferred_dtype = str(server.get("preferred_dtype", "float16")).strip().lower()
    server["preferred_dtype"] = preferred_dtype if preferred_dtype in {"float16", "bf16"} else "float16"

    try:
        timeout_sec = float(server.get("request_timeout_sec", 20))
    except Exception:
        timeout_sec = 20.0
    server["request_timeout_sec"] = max(5.0, min(timeout_sec, 180.0))

    try:
        retry_count = int(server.get("request_retry_count", 2))
    except Exception:
        retry_count = 2
    server["request_retry_count"] = max(0, min(retry_count, 5))

    try:
        retry_backoff = float(server.get("request_retry_backoff_sec", 1.0))
    except Exception:
        retry_backoff = 1.0
    server["request_retry_backoff_sec"] = max(0.1, min(retry_backoff, 10.0))

    raw_level = str(logging_config.get("level", "INFO")).strip().upper()
    logging_config["level"] = raw_level if raw_level in VALID_LOG_LEVELS else "INFO"

    try:
        fps = int(defaults.get("fps", 8))
    except Exception:
        fps = 8
    fps = max(1, min(60, fps))
    defaults["fps"] = fps

    duration_raw = defaults.get("duration_seconds")
    if duration_raw is None:
        try:
            legacy_frames = int(defaults.get("num_frames", 16))
        except Exception:
            legacy_frames = 16
        duration_seconds = max(0.1, float(legacy_frames) / float(max(1, fps)))
    else:
        try:
            duration_seconds = float(duration_raw)
        except Exception:
            duration_seconds = VIDEO_DURATION_SECONDS_DEFAULT
    duration_seconds = max(0.1, min(VIDEO_DURATION_SECONDS_MAX, duration_seconds))
    defaults["duration_seconds"] = duration_seconds
    defaults["num_frames"] = max(1, int(duration_seconds * fps + 0.5))
    return cleaned


def resolve_path(path_like: str, base_dir: Path) -> Path:
    candidate = Path(path_like).expanduser()
    if not candidate.is_absolute():
        candidate = base_dir / candidate
    return candidate.resolve()


def ensure_runtime_dirs(settings: Dict[str, Any], base_dir: Path) -> None:
    for key in ("models_dir", "outputs_dir", "tmp_dir", "logs_dir"):
        resolve_path(str(settings["paths"][key]), base_dir).mkdir(parents=True, exist_ok=True)


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


def load_raw_settings_file(settings_path: Path) -> Optional[Dict[str, Any]]:
    if not settings_path.exists():
        return None
    try:
        payload = json.loads(settings_path.read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else None
    except Exception:
        return None

