# UI Full Validation Summary

- started_at: 2026-02-23T06:05:02.491042+00:00
- ended_at: 2026-02-23T06:05:29.551909+00:00
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
- Model Search (empty query / source switch): passed (6.97s)
  - details: {"result_counts": {"huggingface": 0, "civitai": 30, "all": 30}, "thumbnail_count": 15}
- Settings persistence: passed (2.56s)
  - details: {"guidance_before": "9", "guidance_after_reload": "8.7", "persisted": true}
- Prepare local models: passed (0.51s)
  - details: {"downloaded": ["madebyollin/sdxl-vae-fp16-fix:model_index_added"], "skipped": ["image-to-video model skipped: ali-vilab/i2vgen-xl size=27.74GB > 10.0GB"]}
- Local models UI: passed (1.46s)
  - details: {"rows": 6, "thumbnail_rows": 0}
- Generate Text-to-Image (LoRA/VAE off): failed (0.24s)
  - error captured (see report.json)
- Generate Text-to-Image (LoRA/VAE on): failed (0.20s)
  - error captured (see report.json)
- Generate Image-to-Image (LoRA/VAE off): failed (0.23s)
  - error captured (see report.json)
- Generate Image-to-Image (LoRA/VAE on): failed (0.22s)
  - error captured (see report.json)
- Generate Text-to-Video: failed (0.22s)
  - error captured (see report.json)
- Generate Image-to-Video: failed (0.00s)
  - error captured (see report.json)
- Outputs UI view/delete: passed (2.88s)
  - details: {"count_before": 3, "count_after": 2, "deleted_candidate": "text2video_f7de3b53-d5f4-4cd9-a819-febaac4e051c.mp4 サイズ=129 KB | 更新日時=2026/2/23 14:23:36 | タグ=動画"}
- GPU usage evidence: passed (0.06s)
  - details: {"device": "cuda", "cuda_available": true, "rocm_available": true, "has_pipeline_load_log": false, "has_cuda_log": false}