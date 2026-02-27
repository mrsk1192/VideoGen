# Breaking Changes

## Summary

Video stack was reworked around a unified model registry + adapter layer.
This is a functional redesign, not a cosmetic refactor.

## API changes

1. Added `GET /api/video/runtime`
   - exposes video-oriented hardware profile and effective load policy
2. Added `GET /api/video/models`
   - exposes model spec registry and runtime-derived support levels
3. Runtime surface now distinguishes base runtime diagnostics vs video-runtime diagnostics.

## Behavior changes

1. T2V/I2V pipeline selection is now model-family based (Wan/CogVideoX/TextToVideoSD/etc).
2. Non-Diffusers local video directories are auto-resolved to `*-Diffusers` repository candidates.
3. Video workers now use adapter-based input preparation; unsupported families fail early with explicit messages.

## Settings defaults impacting runtime

1. CPU memory cap for video load is reduced to safe-side default (`4GB` cap path).
2. ROCm pre-import environment for video stability is enforced before `torch` import.

## UI impact

1. Video generation now validates model support against `/api/video/models` prior to submit.
2. Unsupported model/task pairs fail fast in client with explanatory error message.
3. Model preview cards now display runtime support badge (`ready/limited/requires_patch/not_supported`).
