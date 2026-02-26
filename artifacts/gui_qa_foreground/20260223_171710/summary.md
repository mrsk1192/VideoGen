# GUI QA Foreground Summary

- started_at: 2026-02-23T08:17:10.399439+00:00
- ended_at: 2026-02-23T08:27:14.146411+00:00
- base_url: http://127.0.0.1:55349
- driver_mode: edge-headful

## Step Results
- 01_blank_model_search: passed (2.89s)
  - details: {"rows": 0, "thumbnails": 0}
- 02_ensure_models_via_gui_download: passed (16.73s)
  - details: {"actions": [{"repo": "hf-internal-testing/tiny-stable-diffusion-pipe", "status": "already_downloaded_or_not_listed"}, {"repo": "latent-consistency/lcm-lora-sdxl", "status": "already_downloaded_or_not_listed"}, {"repo": "madebyollin/sdxl-vae-fp16-fix", "status": "already_downloaded_or_not_listed"}, {"repo": "ali-vilab/text-to-video-ms-1.7b", "status": "already_downloaded_or_not_listed"}, {"repo": "ali-vilab/i2vgen-xl", "status": "already_downloaded_or_not_listed"}]}
- 03_apply_local_models: passed (4.36s)
  - details: {"applied": {"text-to-image": true, "image-to-image": true, "text-to-video": true, "image-to-video": true}}
- 04_settings_persistence: passed (2.55s)
  - details: {"before": "9", "after_reload": "8.7", "persisted": true}
- 05_generate_t2i_lora_vae_off: failed (539.04s)
  - error captured in report.json
- 06_generate_t2i_lora_vae_on: failed (0.65s)
  - error captured in report.json
- 07_generate_i2i_lora_vae_off: failed (0.17s)
  - error captured in report.json
- 08_generate_i2i_lora_vae_on: failed (0.15s)
  - error captured in report.json
- 09_generate_t2v: failed (0.15s)
  - error captured in report.json
- 10_generate_i2v: failed (0.06s)
  - error captured in report.json
- 11_outputs_view_only: passed (1.82s)
  - details: {"count": 0}
- 12_gpu_evidence: passed (0.02s)
  - details: {"runtime_info": "実行環境の取得に失敗: Failed to fetch", "runtime_has_cuda": false, "runtime_has_rocm": false, "server_has_pipeline_cuda": true}