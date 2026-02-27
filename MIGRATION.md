# Migration Guide (旧 -> 新)

## 1) 設定ファイル移行

既存 `data/settings.json` はそのまま読み込み可能です。  
起動時に不足キーは自動補完されます。

### 新規推奨キー

```json
{
  "server": {
    "preferred_dtype": "float16",
    "gpu_max_concurrency": 1,
    "allow_software_video_fallback": false,
    "request_timeout_sec": 20,
    "request_retry_count": 2,
    "request_retry_backoff_sec": 1.0
  },
  "storage": {
    "cleanup_enabled": true,
    "cleanup_max_age_days": 7,
    "cleanup_max_outputs_count": 200,
    "cleanup_max_tmp_count": 300,
    "cleanup_max_cache_size_gb": 30
  }
}
```

## 2) API移行

### 追加

- `GET /api/runtime`
  - ROCm/torch/dtype/VRAM/同時実行設定を確認可能
- `POST /api/cleanup`
  - `outputs/tmp/cache` の整理を手動実行

### タスク状態

`/api/tasks*` の `status` に `cancelled` が追加されました。  
クライアント側のステータス表示分岐を追加してください。

## 3) UI移行ポイント

1. Durationのラベル・説明は i18n キー化されました。
   - `labelDurationSeconds`
   - `helpDurationSeconds`
2. 追加UI:
   - 現在タスクのキャンセルボタン
   - Downloads内キャンセル
   - Task Log 折りたたみ
   - Cleanup実行ボタン

## 4) 運用移行ポイント（ROCm）

1. `preferred_dtype=bf16` を使う場合、`/api/runtime` の `dtype` と `dtype_warning` を確認してください。
2. OOMが出る場合は以下を優先調整:
   - Duration / Frames / Width / Height / Steps
   - `server.gpu_max_concurrency` を 1 に固定
3. AOTriton設定が反映されているか `GET /api/runtime` の `aotriton_mismatch` で確認してください。

