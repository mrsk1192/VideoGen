# Breaking Changes

## 概要

この更新では、ROCm前提の自動ロード最適化（VRAM判定）とタスク可視化を優先し、
設定項目・タスクJSON・runtime診断項目を拡張しています。

## API

1. `GET /api/runtime` のレスポンスを拡張しました。  
   主な追加:
   - `hardware_profile`（GPU名/総VRAM/空きVRAM/RAM）
   - `load_policy_preview`（候補ロード戦略と選択ポリシー）
   - `last_load_policy`（直近で実際に適用されたポリシー）
2. `GET /api/tasks*` のタスクJSONに `step` を追加しました。  
   UI/クライアント側で `step` を未考慮の場合、表示ロジックの追加が必要です。
3. `POST /api/cleanup` を追加済み（前回追加）。  
   今回の storage/settings 拡張と連動します。

## 設定ファイル

`data/settings.json` に以下を追加:

- `server.preferred_dtype`
- `server.vram_gpu_direct_load_threshold_gb`
- `server.enable_device_map_auto`
- `server.enable_model_cpu_offload`
- `server.gpu_max_concurrency`
- `server.allow_software_video_fallback`
- `server.request_timeout_sec`
- `server.request_retry_count`
- `server.request_retry_backoff_sec`
- `storage.cleanup_enabled`
- `storage.cleanup_max_age_days`
- `storage.cleanup_max_outputs_count`
- `storage.cleanup_max_tmp_count`
- `storage.cleanup_max_cache_size_gb`

`server.preferred_dtype` の実運用既定は `bf16` 優先に変更されました  
（未対応GPUでは `float16` に自動フォールバック）。

## UI

1. Settingsにロード戦略制御項目を追加:
   - `VRAM Direct Load Threshold (GB)`
   - `Enable device_map='auto'`
   - `Enable CPU Offload Fallback`
2. ステータス欄に `step + spinner` 表示を追加しました。
3. ロード中メッセージが詳細化されました:
   - `VRAMにロード中`
   - `自動device_mapでロード中`
   - `CPUオフロード有効化中`
4. `Task Log` は `step/message` 更新をより細かく反映します。
