import logging
import os
import threading
import time
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Dict, Optional

from .config import resolve_path

LOG_FILE_MAX_BYTES = 10 * 1024 * 1024
LOG_FILE_BACKUP_COUNT = 5

LOGGER = logging.getLogger("videogen")
LOGGER_LOCK = threading.Lock()
LOGGER_READY = False
CURRENT_LOG_FILE: Optional[Path] = None
CURRENT_LOG_LEVEL: Optional[int] = None
PROCESS_LOG_TIMESTAMP = time.strftime("%Y%m%d_%H%M%S", time.localtime())
LOG_FILE_NAME = f"{PROCESS_LOG_TIMESTAMP}_videogen_pid{os.getpid()}.log"


def get_logs_dir(settings: Dict[str, Any], base_dir: Path) -> Path:
    logs_dir_raw = settings.get("paths", {}).get("logs_dir", "logs")
    return resolve_path(str(logs_dir_raw), base_dir)


def get_log_file_path(settings: Dict[str, Any], base_dir: Path) -> Path:
    return get_logs_dir(settings, base_dir) / LOG_FILE_NAME


def latest_log_file(settings: Dict[str, Any], base_dir: Path) -> Optional[Path]:
    logs_dir = get_logs_dir(settings, base_dir)
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


def setup_logger(settings: Dict[str, Any], base_dir: Path) -> None:
    global LOGGER_READY, CURRENT_LOG_FILE, CURRENT_LOG_LEVEL
    log_file = get_log_file_path(settings, base_dir)
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
