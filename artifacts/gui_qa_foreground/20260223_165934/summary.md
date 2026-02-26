# GUI QA Foreground Summary

- started_at: 2026-02-23T07:59:34.800176+00:00
- ended_at: 2026-02-23T08:07:25.052365+00:00
- base_url: http://127.0.0.1:56009
- driver_mode: edge-headful

## Step Results
- 01_blank_model_search: passed (2.92s)
  - details: {"rows": 0, "thumbnails": 0}
- 02_ensure_models_via_gui_download: passed (17.01s)
  - details: {"actions": [{"repo": "hf-internal-testing/tiny-stable-diffusion-pipe", "status": "already_downloaded_or_not_listed"}, {"repo": "latent-consistency/lcm-lora-sdxl", "status": "already_downloaded_or_not_listed"}, {"repo": "madebyollin/sdxl-vae-fp16-fix", "status": "already_downloaded_or_not_listed"}, {"repo": "ali-vilab/text-to-video-ms-1.7b", "status": "already_downloaded_or_not_listed"}, {"repo": "ali-vilab/i2vgen-xl", "status": "already_downloaded_or_not_listed"}]}
- 03_apply_local_models: passed (4.40s)
  - details: {"applied": {"text-to-image": true, "image-to-image": true, "text-to-video": true, "image-to-video": true}}
- 04_settings_persistence: passed (2.53s)
  - details: {"before": "9", "after_reload": "8.7", "persisted": true}
- 05_generate_t2i_lora_vae_off: passed (16.12s)
  - details: {"task_id": "80b9e8e3-ceff-4207-9ac7-fa086025fba1", "status_text": "task=80b9e8e3-ceff-4207-9ac7-fa086025fba1 | type=テキスト画像 | status=完了 | 100% | 完了", "file": "text2image_80b9e8e3-ceff-4207-9ac7-fa086025fba1.png", "quality": {"width": 512, "height": 512, "mean_rgb": [152.448, 169.108, 162.326], "stddev_rgb": [68.866, 43.757, 32.434], "edge_mean": 20.257}}
- 06_generate_t2i_lora_vae_on: passed (22.30s)
  - details: {"task_id": "47dd9b57-9aeb-454f-8539-766466d4a8c6", "status_text": "task=47dd9b57-9aeb-454f-8539-766466d4a8c6 | type=テキスト画像 | status=完了 | 100% | 完了", "file": "text2image_47dd9b57-9aeb-454f-8539-766466d4a8c6.png", "quality": {"width": 512, "height": 512, "mean_rgb": [132.672, 159.479, 152.081], "stddev_rgb": [111.184, 89.461, 76.877], "edge_mean": 12.619}}
- 07_generate_i2i_lora_vae_off: passed (10.15s)
  - details: {"task_id": "1e5580a7-586a-49c5-b24a-59eaf72cc522", "status_text": "task=1e5580a7-586a-49c5-b24a-59eaf72cc522 | type=画像画像 | status=完了 | 100% | 完了", "file": "image2image_1e5580a7-586a-49c5-b24a-59eaf72cc522.png", "quality": {"width": 512, "height": 512, "mean_rgb": [101.303, 137.449, 68.139], "stddev_rgb": [45.36, 38.895, 23.558], "edge_mean": 3.659}}
- 08_generate_i2i_lora_vae_on: passed (28.34s)
  - details: {"task_id": "3665ae80-1417-41d3-a911-49a57c568450", "status_text": "task=3665ae80-1417-41d3-a911-49a57c568450 | type=画像画像 | status=完了 | 100% | 完了", "file": "image2image_3665ae80-1417-41d3-a911-49a57c568450.png", "quality": {"width": 512, "height": 512, "mean_rgb": [100.478, 136.555, 69.652], "stddev_rgb": [47.077, 41.596, 23.884], "edge_mean": 4.185}}
- 09_generate_t2v: passed (316.60s)
  - details: {"task_id": "29dee242-41c4-4e72-806b-b19be716aff7", "status_text": "task=29dee242-41c4-4e72-806b-b19be716aff7 | type=テキスト動画 | status=完了 | 100% | 完了", "file": "text2video_29dee242-41c4-4e72-806b-b19be716aff7.mp4", "quality": {"size_mb": 0.038, "first_frame_shape": [256, 256, 3], "first_frame_std": 35.08}}
- 10_generate_i2v: passed (10.14s)
  - details: {"task_id": "3006ae28-72a5-4ae9-b186-ec234a5dd0e6", "status_text": "task=3006ae28-72a5-4ae9-b186-ec234a5dd0e6 | type=画像動画 | status=完了 | 100% | 完了", "file": "image2video_3006ae28-72a5-4ae9-b186-ec234a5dd0e6.mp4", "quality": {"size_mb": 0.25, "first_frame_shape": [512, 512, 3], "first_frame_std": 34.483}}
- 11_outputs_view_only: passed (1.67s)
  - details: {"count": 6}
- 12_gpu_evidence: passed (0.01s)
  - details: {"runtime_info": "device=cuda | cuda=true | rocm=true | diffusers=true | torch=2.9.1+rocmsdk20260116", "runtime_has_cuda": true, "runtime_has_rocm": true, "server_has_pipeline_cuda": true}