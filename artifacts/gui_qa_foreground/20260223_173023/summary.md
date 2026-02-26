# GUI QA Foreground Summary

- started_at: 2026-02-23T08:30:23.583342+00:00
- ended_at: 2026-02-23T08:35:40.875345+00:00
- base_url: http://127.0.0.1:64096
- driver_mode: edge-headful

## Step Results
- 01_blank_model_search: passed (2.92s)
  - details: {"rows": 0, "thumbnails": 0}
- 02_ensure_models_via_gui_download: passed (16.90s)
  - details: {"actions": [{"repo": "hf-internal-testing/tiny-stable-diffusion-pipe", "status": "already_downloaded_or_not_listed"}, {"repo": "latent-consistency/lcm-lora-sdxl", "status": "already_downloaded_or_not_listed"}, {"repo": "madebyollin/sdxl-vae-fp16-fix", "status": "already_downloaded_or_not_listed"}, {"repo": "ali-vilab/text-to-video-ms-1.7b", "status": "already_downloaded_or_not_listed"}, {"repo": "ali-vilab/i2vgen-xl", "status": "already_downloaded_or_not_listed"}]}
- 03_apply_local_models: passed (4.36s)
  - details: {"applied": {"text-to-image": true, "image-to-image": true, "text-to-video": true, "image-to-video": true}}
- 04_settings_persistence: passed (2.53s)
  - details: {"before": "9", "after_reload": "8.7", "persisted": true}
- 05_generate_t2i_lora_vae_off: failed (256.58s)
  - error captured in report.json
- 06_generate_t2i_lora_vae_on: failed (0.05s)
  - error captured in report.json
- 07_generate_i2i_lora_vae_off: failed (0.00s)
  - error captured in report.json
- 08_generate_i2i_lora_vae_on: failed (0.00s)
  - error captured in report.json
- 09_generate_t2v: failed (0.00s)
  - error captured in report.json
- 10_generate_i2v: failed (0.00s)
  - error captured in report.json
- 11_outputs_view_only: failed (0.00s)
  - error captured in report.json
- 12_gpu_evidence: failed (0.00s)
  - error captured in report.json