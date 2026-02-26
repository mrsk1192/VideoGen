# UI Full Validation Summary

- started_at: 2026-02-23T06:07:01.683262+00:00
- ended_at: 2026-02-23T06:09:13.481484+00:00
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
- Model Search (empty query / source switch): passed (6.99s)
  - details: {"result_counts": {"huggingface": 0, "civitai": 30, "all": 30}, "thumbnail_count": 0}
- Settings persistence: passed (2.53s)
  - details: {"guidance_before": "9", "guidance_after_reload": "8.7", "persisted": true}
- Prepare local models: passed (0.53s)
  - details: {"downloaded": [], "skipped": ["image-to-video model skipped: ali-vilab/i2vgen-xl size=27.74GB > 10.0GB"]}
- Local models UI: passed (1.44s)
  - details: {"rows": 6, "thumbnail_rows": 0}
- Generate Text-to-Image (LoRA/VAE off): passed (16.93s)
  - details: {"task_id": "9d935715-e7d2-4cfd-a887-af0181888697", "image_file": "text2image_9d935715-e7d2-4cfd-a887-af0181888697.png", "quality": {"width": 512, "height": 512, "mean_rgb": [133.877, 122.163, 122.082], "stddev_rgb": [119.21, 120.914, 120.919], "edge_mean": 19.63}}
- Generate Text-to-Image (LoRA/VAE on): failed (1.02s)
  - error captured (see report.json)
- Generate Image-to-Image (LoRA/VAE off): passed (10.51s)
  - details: {"task_id": "f2ae0c57-9577-4297-90d6-3e867d14da69", "image_file": "image2image_f2ae0c57-9577-4297-90d6-3e867d14da69.png", "quality": {"width": 512, "height": 512, "mean_rgb": [100.917, 133.431, 69.817], "stddev_rgb": [46.382, 39.821, 24.077], "edge_mean": 3.505}}
- Generate Image-to-Image (LoRA/VAE on): failed (1.08s)
  - error captured (see report.json)
- Generate Text-to-Video: passed (56.18s)
  - details: {"task_id": "9e2c16b7-58b7-4100-8eb2-5805e5308bdf", "video_file": "text2video_9e2c16b7-58b7-4100-8eb2-5805e5308bdf.mp4", "quality": {"size_mb": 0.047, "first_frame_shape": [256, 256, 3], "first_frame_std": 26.95}}
- Generate Image-to-Video: failed (0.00s)
  - error captured (see report.json)
- Outputs UI view/delete: passed (2.94s)
  - details: {"count_before": 5, "count_after": 4, "deleted_candidate": "text2video_9e2c16b7-58b7-4100-8eb2-5805e5308bdf.mp4 サイズ=47.7 KB | 更新日時=2026/2/23 15:08:47 | タグ=動画"}
- GPU usage evidence: passed (0.06s)
  - details: {"device": "cuda", "cuda_available": true, "rocm_available": true, "has_pipeline_load_log": true, "has_cuda_log": true}