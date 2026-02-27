# Migration Guide (旧 -> 新)

## 1) 設定ファイル移行

既存 `data/settings.json` はそのまま読み込み可能です。  
起動時に不足キーは自動補完されます。

### 新規推奨キー

```json
{
  "server": {
    "preferred_dtype": "bf16",
    "vram_gpu_direct_load_threshold_gb": 48,
    "enable_device_map_auto": true,
    "enable_model_cpu_offload": true,
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

### 追加 / 拡張

- `GET /api/runtime`
  - ROCm/torch/dtype/VRAM/RAM/同時実行設定を確認可能
  - 新規確認項目:
    - `hardware_profile`
    - `load_policy_preview`
    - `last_load_policy`
- `POST /api/cleanup`
  - `outputs/tmp/cache` の整理を手動実行

### タスクJSON

- `/api/tasks*` の `status` に `cancelled` が追加済み
- 今回 `task.step` が追加されました
  - `model_load_gpu`, `model_load_auto_map`, `inference`, `encode`, `memory_cleanup` など

## 3) UI移行ポイント

1. Durationのラベル・説明は i18n キー化済み。
   - `labelDurationSeconds`
   - `helpDurationSeconds`
2. 追加UI:
   - 現在タスクのキャンセルボタン
   - Downloads内キャンセル
   - Task Log 折りたたみ
   - Cleanup実行ボタン
   - Task Step + スピナー表示
   - ロードポリシー設定（VRAMしきい値、auto map、CPU offload）

## 4) 運用移行ポイント（ROCm）

1. 既定は `preferred_dtype=bf16` です。`/api/runtime` の `dtype` と `dtype_warning` を確認してください。
2. OOMが出る場合は以下を優先調整:
   - Duration / Frames / Width / Height / Steps
   - `server.vram_gpu_direct_load_threshold_gb`（direct load条件）
   - `server.enable_device_map_auto` / `server.enable_model_cpu_offload`
   - `server.gpu_max_concurrency` を 1 に固定
3. AOTriton設定が反映されているか `GET /api/runtime` の `aotriton_mismatch` で確認してください。
