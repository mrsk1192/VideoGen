# GUI QA Foreground Summary

- started_at: 2026-02-23T08:08:31.375865+00:00
- ended_at: 2026-02-23T08:16:13.241420+00:00
- base_url: http://127.0.0.1:60273
- driver_mode: edge-headful

## Step Results
- 01_blank_model_search: passed (2.90s)
  - details: {"rows": 0, "thumbnails": 0}
- 02_ensure_models_via_gui_download: passed (16.76s)
  - details: {"actions": [{"repo": "hf-internal-testing/tiny-stable-diffusion-pipe", "status": "already_downloaded_or_not_listed"}, {"repo": "latent-consistency/lcm-lora-sdxl", "status": "already_downloaded_or_not_listed"}, {"repo": "madebyollin/sdxl-vae-fp16-fix", "status": "already_downloaded_or_not_listed"}, {"repo": "ali-vilab/text-to-video-ms-1.7b", "status": "already_downloaded_or_not_listed"}, {"repo": "ali-vilab/i2vgen-xl", "status": "already_downloaded_or_not_listed"}]}
- 03_apply_local_models: passed (4.38s)
  - details: {"applied": {"text-to-image": true, "image-to-image": true, "text-to-video": true, "image-to-video": true}}
- 04_settings_persistence: passed (3.03s)
  - details: {"before": "9", "after_reload": "8.7", "persisted": true}
- 05_generate_t2i_lora_vae_off: passed (16.12s)
  - details: {"task_id": "08ade3ee-c44e-41f3-82fe-3c000174fdc3", "status_text": "task=08ade3ee-c44e-41f3-82fe-3c000174fdc3 | type=テキスト画像 | status=完了 | 100% | 完了", "file": "text2image_08ade3ee-c44e-41f3-82fe-3c000174fdc3.png", "quality": {"width": 512, "height": 512, "mean_rgb": [146.94, 148.621, 135.951], "stddev_rgb": [99.939, 80.15, 61.655], "edge_mean": 34.522}}
- 06_generate_t2i_lora_vae_on: passed (22.26s)
  - details: {"task_id": "65a0d049-4d43-48d2-8f16-591aa478a45a", "status_text": "task=65a0d049-4d43-48d2-8f16-591aa478a45a | type=テキスト画像 | status=完了 | 100% | 完了", "file": "text2image_65a0d049-4d43-48d2-8f16-591aa478a45a.png", "quality": {"width": 512, "height": 512, "mean_rgb": [116.525, 131.827, 123.294], "stddev_rgb": [111.672, 100.195, 86.165], "edge_mean": 34.218}}
- 07_generate_i2i_lora_vae_off: passed (13.15s)
  - details: {"task_id": "15d853dd-3343-4612-adc1-c2e6bb8aa936", "status_text": "task=15d853dd-3343-4612-adc1-c2e6bb8aa936 | type=画像画像 | status=完了 | 100% | 完了", "file": "image2image_15d853dd-3343-4612-adc1-c2e6bb8aa936.png", "quality": {"width": 512, "height": 512, "mean_rgb": [98.824, 133.928, 67.941], "stddev_rgb": [50.264, 44.075, 24.684], "edge_mean": 3.708}}
- 08_generate_i2i_lora_vae_on: passed (28.38s)
  - details: {"task_id": "6531a6fc-0724-4873-8c89-11031f625629", "status_text": "task=6531a6fc-0724-4873-8c89-11031f625629 | type=画像画像 | status=完了 | 100% | 完了", "file": "image2image_6531a6fc-0724-4873-8c89-11031f625629.png", "quality": {"width": 512, "height": 512, "mean_rgb": [102.657, 136.496, 71.886], "stddev_rgb": [49.532, 42.186, 26.18], "edge_mean": 4.392}}
- 09_generate_t2v: passed (303.93s)
  - details: {"task_id": "bd3f75a5-78e9-4748-9669-4ce9143891cf", "status_text": "task=bd3f75a5-78e9-4748-9669-4ce9143891cf | type=テキスト動画 | status=完了 | 100% | 完了", "file": "text2video_bd3f75a5-78e9-4748-9669-4ce9143891cf.mp4", "quality": {"size_mb": 0.036, "first_frame_shape": [256, 256, 3], "first_frame_std": 35.876}}
- 10_generate_i2v: passed (10.38s)
  - details: {"task_id": "abeaf0f5-d681-4628-9341-a0df2490e360", "status_text": "task=abeaf0f5-d681-4628-9341-a0df2490e360 | type=画像動画 | status=完了 | 100% | 完了", "file": "image2video_abeaf0f5-d681-4628-9341-a0df2490e360.mp4", "quality": {"size_mb": 0.249, "first_frame_shape": [512, 512, 3], "first_frame_std": 33.89}}
- 11_outputs_view_only: passed (1.82s)
  - details: {"count": 12}
- 12_gpu_evidence: passed (0.01s)
  - details: {"runtime_info": "device=cuda | cuda=true | rocm=true | diffusers=true | torch=2.9.1+rocmsdk20260116", "runtime_has_cuda": true, "runtime_has_rocm": true, "server_has_pipeline_cuda": true}