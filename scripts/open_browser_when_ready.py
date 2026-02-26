import argparse
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
import webbrowser


def wait_until_ready(url: str, timeout_sec: int, interval_sec: float) -> bool:
    deadline = time.time() + timeout_sec
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=1.0) as resp:
                code = int(getattr(resp, "status", 0) or 0)
                if 200 <= code < 500:
                    return True
        except (urllib.error.URLError, TimeoutError, OSError):
            pass
        time.sleep(interval_sec)
    return False


def open_url(url: str) -> bool:
    # Prefer Windows shell association.
    try:
        if hasattr(os, "startfile"):
            os.startfile(url)  # type: ignore[attr-defined]
            return True
    except Exception:
        pass
    try:
        if webbrowser.open(url, new=2):
            return True
    except Exception:
        return False
    # Final fallback for Windows environments where associations are strict.
    if os.name == "nt":
        try:
            subprocess.Popen(["cmd", "/c", "start", "", url], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return True
        except Exception:
            pass
    return False


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--interval", type=float, default=0.5)
    args = parser.parse_args()

    if not wait_until_ready(args.url, timeout_sec=args.timeout, interval_sec=args.interval):
        return 2
    if not open_url(args.url):
        return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
