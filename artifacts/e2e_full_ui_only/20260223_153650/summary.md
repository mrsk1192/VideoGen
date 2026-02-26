# UI Screen-Only Validation Summary

- started_at: 2026-02-23T06:36:50.095704+00:00
- ended_at: 2026-02-23T06:38:50.355874+00:00
- driver_mode: edge-headful

## Step results
- Model Search (blank query): passed (2.45s)
  - details: {"rows": 0, "thumbs": 0}
- Model Download via UI: passed (4.56s)
  - details: {"downloads": [{"repo": "latent-consistency/lcm-lora-sdxl", "status": "already_downloaded_or_not_listed"}, {"repo": "madebyollin/sdxl-vae-fp16-fix", "status": "already_downloaded_or_not_listed"}]}
- Settings Persistence via UI: passed (2.54s)
  - details: {"persisted": true, "guidance_before": "9", "guidance_after_reload": "8.7", "runtime_info": "device=cuda | cuda=true | rocm=true | diffusers=true | torch=2.9.1+rocmsdk20260116"}
- Apply Local Models via UI: passed (3.37s)
  - details: {"rows": 6, "applied": {"text-to-image": true, "image-to-image": true, "text-to-video": true, "image-to-video": false}}
- Generate T2I (LoRA/VAE OFF): passed (15.63s)
  - details: {"task_id": "106a944e-f702-4e0a-abea-a92547c64f9a", "status_text": "task=106a944e-f702-4e0a-abea-a92547c64f9a | type=テキスト画像 | status=完了 | 100% | 完了", "image_file": "text2image_106a944e-f702-4e0a-abea-a92547c64f9a.png", "quality": {"width": 512, "height": 512, "mean_rgb": [123.752, 132.74, 126.868], "stddev_rgb": [101.536, 92.208, 91.732], "edge_mean": 39.983}}
- Generate T2I (LoRA/VAE ON): passed (29.74s)
  - details: {"task_id": "d9c3cae4-fb5b-4758-89ad-989a1146d4fd", "status_text": "task=d9c3cae4-fb5b-4758-89ad-989a1146d4fd | type=テキスト画像 | status=完了 | 100% | 完了", "image_file": "text2image_d9c3cae4-fb5b-4758-89ad-989a1146d4fd.png", "quality": {"width": 512, "height": 512, "mean_rgb": [134.764, 132.833, 135.468], "stddev_rgb": [90.591, 90.854, 89.101], "edge_mean": 124.461}}
- Generate I2I (LoRA/VAE OFF): passed (10.10s)
  - details: {"task_id": "1e87009b-bb77-4b34-97b8-ba591308cb14", "status_text": "task=1e87009b-bb77-4b34-97b8-ba591308cb14 | type=画像画像 | status=完了 | 100% | 完了", "image_file": "image2image_1e87009b-bb77-4b34-97b8-ba591308cb14.png", "quality": {"width": 512, "height": 512, "mean_rgb": [129.85, 127.928, 126.869], "stddev_rgb": [91.814, 92.086, 91.732], "edge_mean": 40.084}}
- Generate I2I (LoRA/VAE ON): passed (3.97s)
  - details: {"task_id": "664fd210-9921-4201-aa34-a59b3a1ef130", "status_text": "task=664fd210-9921-4201-aa34-a59b3a1ef130 | type=画像画像 | status=完了 | 100% | 完了", "image_file": "image2image_664fd210-9921-4201-aa34-a59b3a1ef130.png", "quality": {"width": 512, "height": 512, "mean_rgb": [134.764, 132.833, 135.468], "stddev_rgb": [90.591, 90.854, 89.101], "edge_mean": 124.461}}
- Generate T2V: passed (37.15s)
  - details: {"task_id": "f91ebecf-bbdb-46f4-9988-1b39a63d1a32", "status_text": "task=f91ebecf-bbdb-46f4-9988-1b39a63d1a32 | type=テキスト動画 | status=完了 | 100% | 完了", "video_file": "text2video_f91ebecf-bbdb-46f4-9988-1b39a63d1a32.mp4", "quality": {"size_mb": 0.086, "first_frame_shape": [256, 256, 3], "first_frame_std": 75.936}}
- Generate I2V: failed (0.06s)
  - error captured in report.json
- Outputs View/Delete via UI: passed (2.73s)
  - details: {"count_before": 13, "count_after": 12, "deleted_candidate": "text2video_f91ebecf-bbdb-46f4-9988-1b39a63d1a32.mp4 サイズ=87.7 KB | 更新日時=2026/2/23 15:38:43 | タグ=動画"}
- GPU Evidence: passed (0.01s)
  - details: {"runtime_info_text": "device=cuda | cuda=true | rocm=true | diffusers=true | torch=2.9.1+rocmsdk20260116", "runtime_has_device_cuda": true, "runtime_has_rocm_true": true, "server_log_has_pipeline_cuda": true}