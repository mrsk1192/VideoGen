# Migration Guide (old -> redesigned video stack)

## 1) New video diagnostics endpoints

- `GET /api/video/runtime`
  - video-specific hardware profile and selected load policy
- `GET /api/video/models`
  - model registry entries + effective support level for current runtime
  - new `task_support` field per model (`text-to-video`/`image-to-video`)

Recommended:
1. call `/api/video/runtime` at startup
2. call `/api/video/models` and filter model selection by `effective_support_level`

## 2) Model selection behavior change

- Local non-Diffusers video model directories (missing `model_index.json`) are no longer treated as hard stop by default.
- Runtime attempts fallback resolution to `*-Diffusers` repo IDs.

Operationally:
1. if local folder has no `model_index.json`, keep metadata (`videogen_meta.json` / `model_meta.json`)
2. ensure matching `*-Diffusers` repo is downloadable

## 3) Adapter-based generation

T2V/I2V generation now routes through adapter abstraction.

- Ready families: `TextToVideoSD`, `Wan`, `CogVideoX`
- Limited families: `LTX-Video`, `HunyuanVideo`, `AnimateDiff`
- Requires patch: `SanaVideo`

If you previously hardcoded kwargs per pipeline in external scripts, move those customizations behind adapter logic.

## 4) Settings considerations

Video loading is now more aggressive on ROCm-safe defaults:

- GPU-first load policy remains default on large VRAM
- CPU memory cap path is tighter for video workload safety

If migration requires more CPU slack, explicitly override server memory caps in settings.
