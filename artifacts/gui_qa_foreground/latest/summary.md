# GUI QA Foreground Summary

- started_at: 2026-02-23T16:17:24.015884+00:00
- ended_at: 2026-02-23T16:47:41.601353+00:00
- base_url: http://127.0.0.1:54007
- driver_mode: edge-headful

## Step Results
- 01_blank_model_search: passed (2.90s)
  - details: {"rows": 0, "thumbnails": 0}
- 02_ensure_models_via_gui_download: passed (20.32s)
  - details: {"actions": [{"repo": "SG161222/RealVisXL_V5.0", "status": "already_downloaded_or_not_listed"}, {"repo": "latent-consistency/lcm-lora-sdxl", "status": "already_downloaded_or_not_listed"}, {"repo": "madebyollin/sdxl-vae-fp16-fix", "status": "already_downloaded_or_not_listed"}, {"repo": "damo-vilab/text-to-video-ms-1.7b", "status": "already_downloaded_or_not_listed"}, {"repo": "ali-vilab/text-to-video-ms-1.7b", "status": "already_downloaded_or_not_listed"}, {"repo": "ali-vilab/i2vgen-xl", "status": "already_downloaded_or_not_listed"}]}
- 03_apply_local_models: passed (4.36s)
  - details: {"applied": {"text-to-image": true, "image-to-image": true, "text-to-video": true, "image-to-video": true}}
- 04_settings_persistence: passed (3.18s)
  - details: {"before": "9", "after_reload": "8.7", "persisted": true}
- 05_generate_t2i_lora_vae_off: passed (304.97s)
  - details: {"task_id": "eefeb8c5-409f-4451-a83c-dadf98d4e14d", "status_text": "status=completed progress=1.0", "selected_model": "C:\\AI\\VideoGen\\models\\SG161222--RealVisXL_V5.0", "file": "text2image_eefeb8c5-409f-4451-a83c-dadf98d4e14d.png", "quality": {"width": 768, "height": 768, "mean_rgb": [89.888, 125.618, 159.742], "stddev_rgb": [67.132, 61.106, 58.409], "edge_mean": 11.779}}
- 06_generate_t2i_lora_vae_on: passed (242.60s)
  - details: {"task_id": "4699996c-680b-4520-9828-5b9207bc5de4", "status_text": "task=4699996c-680b-4520-9828-5b9207bc5de4 | type=テキスト画像 | status=完了 | 100% | 完了", "selected_model": "C:\\AI\\VideoGen\\models\\SG161222--RealVisXL_V5.0", "file": "text2image_4699996c-680b-4520-9828-5b9207bc5de4.png", "quality": {"width": 768, "height": 768, "mean_rgb": [85.192, 125.025, 157.789], "stddev_rgb": [85.419, 70.165, 63.26], "edge_mean": 13.045}}
- 07_generate_i2i_lora_vae_off: passed (49.46s)
  - details: {"task_id": "84fe032b-b6dc-4b08-a656-c0e118abf431", "status_text": "task=84fe032b-b6dc-4b08-a656-c0e118abf431 | type=画像画像 | status=完了 | 100% | 完了", "selected_model": "C:\\AI\\VideoGen\\models\\SG161222--RealVisXL_V5.0", "file": "image2image_84fe032b-b6dc-4b08-a656-c0e118abf431.png", "quality": {"width": 768, "height": 768, "mean_rgb": [117.595, 115.844, 106.8], "stddev_rgb": [65.262, 45.004, 28.997], "edge_mean": 5.927}}
- 08_generate_i2i_lora_vae_on: passed (41.39s)
  - details: {"task_id": "70fb331a-03b8-4481-82cc-6a05c88f2bd1", "status_text": "status=completed progress=1.0", "selected_model": "C:\\AI\\VideoGen\\models\\SG161222--RealVisXL_V5.0", "file": "image2image_70fb331a-03b8-4481-82cc-6a05c88f2bd1.png", "quality": {"width": 768, "height": 768, "mean_rgb": [113.455, 121.313, 111.779], "stddev_rgb": [84.903, 61.35, 48.395], "edge_mean": 12.741}}
- 09_generate_t2v: failed (1033.57s)
  - error captured in report.json
- 10_generate_i2v: passed (50.70s)
  - details: {"task_id": "8b8ad58f-d256-43d9-a191-213173099536", "status_text": "status=completed progress=1.0", "selected_model": "C:\\AI\\VideoGen\\models\\ali-vilab--i2vgen-xl", "requested_duration_sec": 5.0, "file": "", "quality": {}}
- 11_outputs_view_only: passed (1.84s)
  - details: {"count": 26}
- 12_gpu_evidence: passed (0.01s)
  - details: {"runtime_info": "device=cuda | cuda=true | rocm=true | diffusers=true | torch=2.9.1+rocmsdk20260116", "runtime_has_cuda": true, "runtime_has_rocm": true, "server_has_pipeline_cuda": true}