# VideoGen Architecture Notes

## Scope of this refactor

- Introduced `videogen/` package for core responsibilities:
  - `videogen/config.py`: settings defaults/sanitization/store/path helpers.
  - `videogen/runtime.py`: pre-`torch` env bootstrap and runtime/device diagnostics.
  - `videogen/tasks.py`: centralized task lifecycle and cancellation model.
  - `videogen/http.py`: retry/timeout helpers for provider and download requests.
  - `videogen/models/*`: provider integration wrappers.
- Kept public API paths and request/response shapes backward compatible.
- Kept `main.py` compatibility surface (tests and existing imports still work).

## Key decisions

### 1) GPU concurrency control

- Generation workloads are serialized with a process-wide semaphore.
- Config: `server.gpu_max_concurrency` (default `1`, range `1..8`).
- Applied to text/image generation workers to reduce ROCm instability and out-of-memory spikes from concurrent pipelines.

### 2) ROCm env consistency (AOTriton + allocator)

- `main.py` now bootstraps env before importing `torch`.
- `TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL` is derived from persisted settings when absent.
- Allocator env (`PYTORCH_ALLOC_CONF` / `PYTORCH_CUDA_ALLOC_CONF`) is normalized at startup.
- `/api/runtime` exposes effective env + mismatch warnings to detect uvicorn direct-launch misconfiguration.
- `/api/runtime` now also reports selected dtype/bf16 capability/device name/gpu count/concurrency policy.

### 3) Device/dtype policy unification

- `get_device_and_dtype()` now follows settings:
  - `server.require_gpu`
  - `server.allow_cpu_fallback`
  - `server.preferred_dtype` (`float16` / `bf16`)
- If `bf16` is requested but unsupported, runtime falls back to `float16` on GPU.
- `/api/runtime` reports selected dtype and bf16 capability.
- 推論本体は `torch.inference_mode()` + `torch.autocast(device_type="cuda", dtype=...)` を共通コンテキストで適用。

### 4) Task cancellation model

- Added `POST /api/tasks/cancel`.
- `TaskManager` stores `cancel_requested` and exposes cancellation checks.
- Long-running loops (downloads/framepack loops/step callbacks) call cancellation checks and transition to `cancelled`.
- HF snapshot download cannot always be interrupted once started; this is logged explicitly.

### 5) Video encode fallback policy

- Default behavior remains hardware-first (AMF) for diagnostics and expected performance.
- Opt-in setting: `server.allow_software_video_fallback`.
- When enabled, fallback encoder is `libx264` via ffmpeg/imageio if AMF initialization fails.

### 6) Storage lifecycle and cleanup

- Added `POST /api/cleanup`.
- Policy source is `settings.storage.*`:
  - max age (days)
  - max file count (outputs/tmp)
  - max HF cache size
- Cleanup runs on explicit user action from UI; policy is deterministic and logs removed counts.

### 7) Network I/O hardening

- External calls now use `httpx` with timeout/retry policy from settings:
  - `server.request_timeout_sec`
  - `server.request_retry_count`
  - `server.request_retry_backoff_sec`
- Download streaming reports progress and supports cancellation checks.
- Compatibility hooks were kept for existing tests that monkeypatch `urlopen`.

## Runtime diagnostics endpoint

- `GET /api/runtime` (also `/api/system/info` compatibility path) includes:
  - torch versions (`torch`, HIP)
  - CUDA/ROCm availability
  - selected device/dtype and bf16 support
  - GPU memory snapshot (when available)
  - AOTriton and allocator env
  - policy mismatch warnings

## Frontend UX improvements

- Added cancel button for active task and per-download cancel in Downloads widget.
- Added loading indicators for model search/detail/local model loading.
- Added polling backoff (`setTimeout`) for tasks/download lists to reduce unnecessary traffic under failures.
- Generation buttons are disabled while generation task is active.
