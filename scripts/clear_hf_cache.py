import argparse
import os
import shutil
import sys
from pathlib import Path
from typing import Iterable, Set


def normalize(path_like: str) -> Path:
    return Path(path_like).expanduser().resolve()


def gather_candidates() -> Set[Path]:
    candidates: Set[Path] = set()

    hf_home_env = os.environ.get("HF_HOME", "").strip()
    if hf_home_env:
        hf_home = normalize(hf_home_env)
    else:
        hf_home = normalize(str(Path.home() / ".cache" / "huggingface"))

    # Explicit env vars take precedence.
    hub_cache_env = os.environ.get("HUGGINGFACE_HUB_CACHE", "").strip()
    if hub_cache_env:
        candidates.add(normalize(hub_cache_env))
    else:
        candidates.add(hf_home / "hub")

    transformers_cache_env = os.environ.get("TRANSFORMERS_CACHE", "").strip()
    if transformers_cache_env:
        candidates.add(normalize(transformers_cache_env))
    else:
        candidates.add(hf_home / "transformers")

    # Other known Hugging Face cache subdirs under HF_HOME.
    for name in ("assets", "xet", "modules"):
        candidates.add(hf_home / name)

    # Resolve library defaults when available.
    try:
        from huggingface_hub import constants as hf_constants  # type: ignore

        for attr in ("HUGGINGFACE_HUB_CACHE", "HF_ASSETS_CACHE"):
            value = getattr(hf_constants, attr, None)
            if isinstance(value, str) and value.strip():
                candidates.add(normalize(value))
    except Exception:
        pass

    try:
        from transformers.utils import TRANSFORMERS_CACHE  # type: ignore

        if isinstance(TRANSFORMERS_CACHE, str) and TRANSFORMERS_CACHE.strip():
            candidates.add(normalize(TRANSFORMERS_CACHE))
    except Exception:
        pass

    return {path for path in candidates if str(path).strip()}


def is_safe_target(path: Path) -> bool:
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


def clear_paths(paths: Iterable[Path], dry_run: bool = False) -> int:
    failures = 0
    for path in sorted(set(paths), key=lambda p: str(p).lower()):
        if not is_safe_target(path):
            print(f"[WARN] Skip unsafe path: {path}")
            continue
        if not path.exists():
            print(f"[INFO] Not found (skip): {path}")
            continue
        if dry_run:
            print(f"[INFO] Would remove: {path}")
            continue
        try:
            shutil.rmtree(path)
            print(f"[INFO] Removed: {path}")
        except Exception as exc:
            failures += 1
            print(f"[WARN] Failed to remove {path}: {exc}")
    return failures


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="Print targets without deleting them.")
    args = parser.parse_args()

    print("[INFO] Clearing Hugging Face caches...")
    candidates = gather_candidates()
    if not candidates:
        print("[INFO] No cache candidates found.")
        return 0
    failures = clear_paths(candidates, dry_run=args.dry_run)
    if failures:
        print(f"[WARN] Cache clear finished with {failures} failure(s).")
        return 1
    print("[INFO] Cache clear finished.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
