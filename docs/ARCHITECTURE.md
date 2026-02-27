# VideoGen Architecture (Video Generation Redesign)

## Why redesign

T2V/I2V path was previously hard-wired to a few pipeline assumptions and grew via incremental patches.  
That made ROCm behavior hard to reason about, and model-specific differences leaked into workers/UI.

## New core design

### 1) Video model registry

- Added `videogen/video/registry.py`
- Defines `VideoModelSpec` with:
  - model identity / task kind (`t2v`, `i2v`, `hybrid`)
  - required inputs
  - recommended dtype
  - ROCm notes
  - support level (`ready`, `limited`, `requires_patch`, `not_supported`)
- Registered targets:
  - TextToVideoSDPipeline
  - WanPipeline
  - CogVideoX
  - LTX-Video
  - HunyuanVideo
  - SanaVideo
  - AnimateDiff

### 2) Unified adapter layer

- Added `videogen/video/adapters.py`
- Introduced `VideoPipelineAdapter` interface:
  - `load()`
  - `prepare_inputs()`
  - `extract_frames()`
- Implemented concrete adapter behavior for:
  - TextToVideoSD (ready)
  - Wan (ready)
  - CogVideoX (ready)
- Remaining families are surfaced as `limited`/`requires_patch` and fail with explicit error guidance.

### 3) Hardware/load-policy abstraction

- Added `videogen/video/foundation.py`
- Introduced:
  - `VideoHardwareProfile`
  - `VideoLoadPolicy`
- Runtime API derives safe defaults per VRAM tier and exposes effective policy.

### 4) Main pipeline integration

- `main.py` now:
  - dynamically resolves pipeline loader by model family
  - supports optional diffusers classes without hard import failure
  - attaches source metadata to loaded pipeline (`_videogen_source`)
  - uses adapter-based request payload generation in both T2V and I2V workers

## ROCm-specific decisions

- ROCm allocator settings are forced before `torch` import.
- 96GB-class profile uses aggressive GPU-first load policy and minimal CPU memory cap.
- GPU generation default concurrency remains safe-side (`1`).
- Worker progress explicitly includes runtime diagnostics stage.

## APIs for video stack

- `GET /api/video/runtime`
  - returns `hardware_profile`, `load_policy`, and base runtime snapshot
- `GET /api/video/models`
  - returns registry + environment-resolved `effective_support_level`
  - includes `task_support` map (`text-to-video` / `image-to-video`) per model

## Current support matrix (effective target)

- `ready`: TextToVideoSD, Wan, CogVideoX
- `limited`: LTX-Video, HunyuanVideo, AnimateDiff
- `requires_patch`: SanaVideo

This matrix is generated at runtime from installed diffusers classes and GPU/runtime state.
