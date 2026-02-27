# Breaking Changes

## 概要

この更新では、ROCm運用性・UX・保守性を優先し、設定項目とAPI/UIの一部仕様を変更しました。

## API

1. `POST /api/cleanup` を追加しました。  
   ストレージ整理を手動実行する正式APIです。

2. `GET /api/runtime` のレスポンスを拡張しました。  
   追加フィールド:
   - `dtype`
   - `dtype_warning`
   - `gpu_name`
   - `gpu_device_count`
   - `gpu_max_concurrency`
   - `gpu_max_concurrency_effective`

3. タスク `status` に `cancelled` を追加しました。  
   既存クライアントで `queued/running/completed/error` のみを想定している場合は対応が必要です。

## 設定ファイル

`data/settings.json` に以下を追加:

- `server.preferred_dtype`
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

## UI

1. 生成中は送信ボタンが自動で無効化されます。  
2. 現在タスクの `Cancel Task` ボタンを追加しました。  
3. Downloadsウィジェット内でダウンロードタスクを個別キャンセル可能になりました。  
4. Durationラベルは多言語キー化され、説明文がヘルプに統合されました。  
5. Task Log（折りたたみ）を追加しました。  

