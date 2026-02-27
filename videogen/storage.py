import shutil
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

from .config import resolve_path


def _iter_files(directory: Path) -> Iterable[Path]:
    if not directory.exists():
        return []
    return [entry for entry in directory.iterdir() if entry.is_file()]


def _directory_size_bytes(root: Path) -> int:
    total = 0
    if not root.exists():
        return 0
    for path in root.rglob("*"):
        if path.is_file():
            total += int(path.stat().st_size)
    return total


def _prune_by_age_and_count(
    files: list[Path],
    *,
    max_age_days: int,
    max_count: int,
) -> list[Path]:
    now = time.time()
    max_age_sec = max(1, int(max_age_days)) * 24 * 60 * 60
    ordered = sorted(files, key=lambda p: p.stat().st_mtime if p.exists() else 0)
    remove_targets: list[Path] = []
    for path in ordered:
        age_sec = max(0.0, now - float(path.stat().st_mtime))
        if age_sec > max_age_sec:
            remove_targets.append(path)
    keep = [path for path in ordered if path not in remove_targets]
    overflow = max(0, len(keep) - max(1, int(max_count)))
    if overflow > 0:
        remove_targets.extend(keep[:overflow])
    return remove_targets


def _remove_files(paths: Iterable[Path]) -> list[str]:
    removed: list[str] = []
    for path in paths:
        try:
            if path.exists() and path.is_file():
                path.unlink(missing_ok=True)
                removed.append(str(path))
        except Exception:
            continue
    return removed


def _cleanup_hf_cache(cache_paths: list[Path], max_cache_size_bytes: int) -> Dict[str, Any]:
    before = sum(_directory_size_bytes(path) for path in cache_paths if path.exists())
    if before <= max_cache_size_bytes:
        return {
            "cache_size_before": before,
            "cache_size_after": before,
            "removed_cache_paths": [],
        }
    removed_paths: list[str] = []
    # サイズ上限を超えた場合は、古い順にディレクトリを削除して確実に回復する。
    # HFキャッシュは再取得可能なため、安定運用を優先して大胆に刈り込む。
    ordered = sorted(
        [path for path in cache_paths if path.exists()],
        key=lambda p: p.stat().st_mtime if p.exists() else 0,
    )
    current = before
    for target in ordered:
        if current <= max_cache_size_bytes:
            break
        size = _directory_size_bytes(target)
        try:
            if target.is_dir():
                shutil.rmtree(target, ignore_errors=False)
            else:
                target.unlink(missing_ok=True)
            removed_paths.append(str(target))
            current = max(0, current - size)
        except Exception:
            continue
    after = sum(_directory_size_bytes(path) for path in cache_paths if path.exists())
    return {
        "cache_size_before": before,
        "cache_size_after": after,
        "removed_cache_paths": removed_paths,
    }


def run_cleanup(
    *,
    settings: Dict[str, Any],
    base_dir: Path,
    hf_cache_candidates: Optional[list[Path]] = None,
) -> Dict[str, Any]:
    storage = settings.get("storage", {})
    outputs_dir = resolve_path(str(settings.get("paths", {}).get("outputs_dir", "outputs")), base_dir)
    tmp_dir = resolve_path(str(settings.get("paths", {}).get("tmp_dir", "tmp")), base_dir)
    max_age_days = int(storage.get("cleanup_max_age_days", 7))
    max_outputs_count = int(storage.get("cleanup_max_outputs_count", 200))
    max_tmp_count = int(storage.get("cleanup_max_tmp_count", 300))
    max_cache_size_bytes = int(float(storage.get("cleanup_max_cache_size_gb", 30.0)) * (1024**3))

    outputs_files = list(_iter_files(outputs_dir))
    tmp_files = list(_iter_files(tmp_dir))
    output_remove_targets = _prune_by_age_and_count(outputs_files, max_age_days=max_age_days, max_count=max_outputs_count)
    tmp_remove_targets = _prune_by_age_and_count(tmp_files, max_age_days=max_age_days, max_count=max_tmp_count)
    removed_outputs = _remove_files(output_remove_targets)
    removed_tmp = _remove_files(tmp_remove_targets)

    cache_cleanup = {"cache_size_before": 0, "cache_size_after": 0, "removed_cache_paths": []}
    if hf_cache_candidates:
        cache_cleanup = _cleanup_hf_cache(list(hf_cache_candidates), max_cache_size_bytes=max_cache_size_bytes)

    return {
        "status": "ok",
        "removed_outputs": removed_outputs,
        "removed_tmp": removed_tmp,
        "outputs_count_before": len(outputs_files),
        "outputs_count_after": max(0, len(outputs_files) - len(removed_outputs)),
        "tmp_count_before": len(tmp_files),
        "tmp_count_after": max(0, len(tmp_files) - len(removed_tmp)),
        **cache_cleanup,
    }
