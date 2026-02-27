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
  - `server.preferred_dtype` (`bf16` / `float16`)
- ROCm運用では既定を `bf16` とし、未対応環境では `float16` へ自動フォールバック。
- `/api/runtime` reports selected dtype and bf16 capability.
- 推論本体は `torch.inference_mode()` + `torch.autocast(device_type="cuda", dtype=...)` を共通コンテキストで適用。

### 4) HardwareProfile + LoadPolicy 抽象化

- `videogen/model_loader.py` を追加し、以下を分離:
  - `HardwareProfile`: GPU名/総VRAM/空きVRAM/RAM/bf16可否/ROCm判定
  - `LoadPolicy`: `device_map` / `low_cpu_mem_usage` / offload設定 / CPU offload
- ロード戦略は VRAM 閾値（`server.vram_gpu_direct_load_threshold_gb`, 既定48GB）で自動選択:
  - 高VRAM: `device_map={"": "cuda"}` 優先
  - 低VRAM: `device_map="auto"` -> `enable_model_cpu_offload()` -> CPU低メモリロード
- `from_pretrained` 系は `low_cpu_mem_usage=True` を標準適用。

### 5) Task cancellation model

- Added `POST /api/tasks/cancel`.
- `TaskManager` stores `cancel_requested` and exposes cancellation checks.
- Long-running loops (downloads/framepack loops/step callbacks) call cancellation checks and transition to `cancelled`.
- HF snapshot download cannot always be interrupted once started; this is logged explicitly.

### 6) Video encode fallback policy

- Default behavior remains hardware-first (AMF) for diagnostics and expected performance.
- Opt-in setting: `server.allow_software_video_fallback`.
- When enabled, fallback encoder is `libx264` via ffmpeg/imageio if AMF initialization fails.

### 7) Storage lifecycle and cleanup

- Added `POST /api/cleanup`.
- Policy source is `settings.storage.*`:
  - max age (days)
  - max file count (outputs/tmp)
- max HF cache size
- Cleanup runs on explicit user action from UI; policy is deterministic and logs removed counts.

### 8) Network I/O hardening

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
  - `hardware_profile` (VRAM/RAM/GPU名)
  - `load_policy_preview`（候補と選択される方針）
  - `last_load_policy`（直近実適用ポリシー）
  - GPU memory snapshot (when available)
  - AOTriton and allocator env
  - policy mismatch warnings

## Frontend UX improvements

- Added cancel button for active task and per-download cancel in Downloads widget.
- Added loading indicators for model search/detail/local model loading.
- Added task `step` visualization with spinner (`model_load_gpu`, `model_load_auto_map`, `inference`, `encode`, `memory_cleanup`).
- Added polling backoff (`setTimeout`) for tasks/download lists to reduce unnecessary traffic under failures.
- Generation buttons are disabled while generation task is active.
