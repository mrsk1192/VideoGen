# UI Full Validation Summary

- started_at: 2026-02-23T05:58:18.063095+00:00
- ended_at: 2026-02-23T05:59:58.842856+00:00
- driver_mode: edge-headful
- runtime: {"diffusers_ready": true, "cuda_available": true, "rocm_available": true, "device": "cuda", "torch_version": "2.9.1+rocmsdk20260116", "import_error": null}

## Test viewpoints
- Startup and health (`/api/system/info`, server readiness).
- UI navigation across all tabs and multilingual/search controls.
- Settings save and reload persistence.
- Model preparation (local scan + auto download for LoRA/VAE if absent).
- Generation paths: text->image, image->image, text->video, image->video.
- LoRA/VAE on/off comparison on image workflows.
- Outputs view/delete from UI.
- GPU evidence from system info and logs.

## Step results
- Model Search (empty query / source switch): passed (7.00s)
  - details: {"result_counts": {"huggingface": 0, "civitai": 30, "all": 30}, "thumbnail_count": 0}
- Settings persistence: passed (2.55s)
  - details: {"guidance_before": "9", "guidance_after_reload": "8.7", "persisted": true}
- Prepare local models: passed (75.20s)
  - details: {"downloaded": ["latent-consistency/lcm-lora-sdxl", "madebyollin/sdxl-vae-fp16-fix"], "skipped": ["image-to-video model skipped: ali-vilab/i2vgen-xl size=27.74GB > 10.0GB"]}
- Local models UI: passed (1.49s)
  - details: {"rows": 6, "thumbnail_rows": 0}
- Generate Text-to-Image (LoRA/VAE off): failed (0.00s)
  - error captured (see report.json)
- Generate Text-to-Image (LoRA/VAE on): failed (0.00s)
  - error captured (see report.json)
- Generate Image-to-Image (LoRA/VAE off): failed (0.00s)
  - error captured (see report.json)
- Generate Image-to-Image (LoRA/VAE on): failed (0.00s)
  - error captured (see report.json)
- Generate Text-to-Video: failed (0.18s)
  - error captured (see report.json)
- Generate Image-to-Video: failed (0.00s)
  - error captured (see report.json)
- Outputs UI view/delete: failed (2.69s)
  - error captured (see report.json)
- GPU usage evidence: passed (0.06s)
  - details: {"device": "cuda", "cuda_available": true, "rocm_available": true, "has_pipeline_load_log": false, "has_cuda_log": false}