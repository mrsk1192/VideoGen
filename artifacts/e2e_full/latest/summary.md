# UI Full Validation Summary

- started_at: 2026-02-23T06:09:45.928831+00:00
- ended_at: 2026-02-23T06:12:23.598461+00:00
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
- Model Search (empty query / source switch): passed (6.98s)
  - details: {"result_counts": {"huggingface": 0, "civitai": 30, "all": 30}, "thumbnail_count": 15}
- Settings persistence: passed (2.54s)
  - details: {"guidance_before": "9", "guidance_after_reload": "8.7", "persisted": true}
- Prepare local models: passed (0.55s)
  - details: {"downloaded": [], "skipped": ["image-to-video model skipped: ali-vilab/i2vgen-xl size=27.74GB > 10.0GB"]}
- Local models UI: passed (1.45s)
  - details: {"rows": 6, "thumbnail_rows": 0}
- Generate Text-to-Image (LoRA/VAE off): passed (16.41s)
  - details: {"task_id": "231cc1af-4656-42ac-9321-7b605f244b8d", "image_file": "text2image_231cc1af-4656-42ac-9321-7b605f244b8d.png", "quality": {"width": 512, "height": 512, "mean_rgb": [125.108, 123.336, 110.507], "stddev_rgb": [83.443, 70.937, 65.138], "edge_mean": 33.868}}
- Generate Text-to-Image (LoRA/VAE on): passed (40.80s)
  - details: {"task_id": "d139c8b7-7bce-4147-a922-24aeff3436a5", "image_file": "text2image_d139c8b7-7bce-4147-a922-24aeff3436a5.png", "quality": {"width": 512, "height": 512, "mean_rgb": [132.099, 134.569, 131.391], "stddev_rgb": [78.906, 81.327, 80.796], "edge_mean": 133.054}}
- Generate Image-to-Image (LoRA/VAE off): passed (8.85s)
  - details: {"task_id": "a5794f35-ea4e-4350-9eae-8746886fe014", "image_file": "image2image_a5794f35-ea4e-4350-9eae-8746886fe014.png", "quality": {"width": 512, "height": 512, "mean_rgb": [112.247, 107.686, 110.503], "stddev_rgb": [70.095, 63.368, 65.138], "edge_mean": 33.912}}
- Generate Image-to-Image (LoRA/VAE on): passed (4.31s)
  - details: {"task_id": "088a817b-c74a-41b9-bdee-02e9ac5effd0", "image_file": "image2image_088a817b-c74a-41b9-bdee-02e9ac5effd0.png", "quality": {"width": 512, "height": 512, "mean_rgb": [132.099, 134.569, 131.391], "stddev_rgb": [78.906, 81.327, 80.796], "edge_mean": 133.054}}
- Generate Text-to-Video: passed (43.03s)
  - details: {"task_id": "d9b21c88-881a-4da0-96ac-c5a063346dec", "video_file": "text2video_d9b21c88-881a-4da0-96ac-c5a063346dec.mp4", "quality": {"size_mb": 0.081, "first_frame_shape": [256, 256, 3], "first_frame_std": 56.507}}
- Generate Image-to-Video: failed (0.00s)
  - error captured (see report.json)
- Outputs UI view/delete: passed (2.92s)
  - details: {"count_before": 9, "count_after": 8, "deleted_candidate": "text2video_d9b21c88-881a-4da0-96ac-c5a063346dec.mp4 サイズ=82.7 KB | 更新日時=2026/2/23 15:11:59 | タグ=動画"}
- GPU usage evidence: passed (0.06s)
  - details: {"device": "cuda", "cuda_available": true, "rocm_available": true, "has_pipeline_load_log": true, "has_cuda_log": true}