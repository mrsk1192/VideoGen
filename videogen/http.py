import asyncio
import time
from typing import Any, Dict, Optional

import httpx


class HttpRetryError(RuntimeError):
    pass


async def request_json_with_retry(
    *,
    method: str,
    url: str,
    params: Optional[Dict[str, Any]] = None,
    headers: Optional[Dict[str, str]] = None,
    timeout_sec: float = 20.0,
    retry_count: int = 2,
    retry_backoff_sec: float = 1.0,
) -> Dict[str, Any]:
    last_error: Optional[Exception] = None
    timeout = httpx.Timeout(timeout_sec, connect=min(timeout_sec, 10.0))
    for attempt in range(retry_count + 1):
        try:
            async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
                response = await client.request(method=method, url=url, params=params, headers=headers)
                response.raise_for_status()
                payload = response.json()
                if not isinstance(payload, dict):
                    raise HttpRetryError(f"Invalid JSON object response from {url}")
                return payload
        except (httpx.HTTPError, ValueError) as exc:
            last_error = exc
            if attempt >= retry_count:
                break
            await asyncio.sleep(max(0.1, retry_backoff_sec) * (2**attempt))
    raise HttpRetryError(str(last_error or "request failed"))


def stream_download_with_retry(
    *,
    url: str,
    destination_path: str,
    timeout_sec: float = 60.0,
    retry_count: int = 2,
    retry_backoff_sec: float = 1.0,
    chunk_bytes: int = 1024 * 1024,
):
    last_error: Optional[Exception] = None
    timeout = httpx.Timeout(timeout_sec, connect=min(timeout_sec, 10.0))
    for attempt in range(retry_count + 1):
        try:
            with httpx.stream("GET", url, timeout=timeout, follow_redirects=True) as response:
                response.raise_for_status()
                with open(destination_path, "wb") as handle:
                    for chunk in response.iter_bytes(chunk_size=chunk_bytes):
                        if chunk:
                            handle.write(chunk)
                return response.headers
        except (httpx.HTTPError, OSError) as exc:
            last_error = exc
            if attempt >= retry_count:
                break
            time.sleep(max(0.1, retry_backoff_sec) * (2**attempt))
    raise HttpRetryError(str(last_error or "download failed"))
