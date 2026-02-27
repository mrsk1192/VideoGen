# ROCm VideoGen Web App

ROCm 環境で動作する `Text-to-Image` / `Image-to-Image` / `Text-to-Video` / `Image-to-Video` の Web アプリ（開発中）です。

## 機能

- Web GUI からテキスト→動画生成
- Web GUI からテキスト→画像生成
- Web GUI から画像→画像生成
- Web GUI から画像→動画生成
- 各生成機能で複数LoRAを選択して適用可能（共通スケール）
- Text-to-Image / Image-to-Image で VAE 選択可能（ローカルVAEカタログ）
- Hugging Face モデル検索
- CivitAI モデル検索（画像系タスク）
- Model Search のカードUI（Grid/List）、詳細ペイン、ページング
- Model Search の詳細モーダル表示（詳細ボタン）
- 右上 Downloads ウィジェット（進捗一覧・ステータス確認）
- Model Search の詳細表示（説明、タグ、プレビュー、バージョン/ファイル選択）
- Hugging Face / CivitAI のモデルダウンロード
- 検索語なしでもモデル検索可能（タスク別の人気順）
- 検索結果にプレビュー画像がある場合は表示
- 生成モデルはドロップダウンから選択可能
- モデル選択時にサムネイル表示
- サーバサイドでモデルをダウンロード
- `Models` タブからローカルモデルを削除
- 設定値保存 (`data/settings.json`)
- 生成/ダウンロード失敗時の詳細ログ出力（スタックトレース含む）
- 生成ジョブの進捗表示と動画/画像プレビュー
- 動画生成は FramePack 方式（小パック逐次生成 + 逐次エンコード）で実行
- Web GUI 多言語表示（`en`, `ja`, `es`, `fr`, `de`, `it`, `pt`, `ru`, `ar`）
- モデル保存先をフルパスで指定可能
- ローカルモデル一覧にサムネイル表示、系譜ドロップダウン絞り込み、生成タスクへ適用ボタン
- Local Models のツリー表示（Task > Base > Category > Item）と検索フィルタ
- Local Models の再走査（Rescan）と Explorer で場所を開く（Windows）
- リッスンポートを `Settings` 画面から変更可能（次回起動時に反映）

## 前提

- Windows + ROCm 対応 GPU
- Python 3.12 推奨
- ROCm 7.2 を使う場合は、以下のセットアップバッチを使用

## ROCm 7.2 セットアップ

```bash
setup_rocm72.bat
```

`setup_rocm72.bat` は以下を実行します。

- `venv` 作成（Python 3.12）
- ROCm 7.2 SDK wheel インストール
- ROCm 7.2 対応 `torch/torchaudio/torchvision` インストール
- `requirements.txt` インストール

## クリーンvenv手動セットアップ（Windows + ROCm）

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.windows-rocm.txt
```

`torch==2.9.0+rocmsdk20251116` は通常の PyPI にはないため、
`requirements.txt` + `constraints.windows-rocm.txt` だけでは解決できません。

## 起動

```bash
start.bat
```

`start.bat` は `venv\Scripts\python.exe` を優先して起動します。  
`venv` がない場合は `.venv` を作成して起動します。
サーバー起動を検知してからブラウザを自動起動します（無効化: `set AUTO_OPEN_BROWSER=0`）。
Hugging Face キャッシュは `Settings` 画面の `Clear Cache` ボタンから削除できます。

ブラウザで `http://localhost:8000` を開いて利用します。

## 使い方

1. `Settings` タブで保存先ディレクトリやデフォルトモデルを確認・保存
2. `Models` タブでモデルを検索し、`Download` を実行（Hugging Face / CivitAI をサーバ側に保存）
3. `Local Models` タブで Tree/Flat を選択
   - Tree: `T2I/I2I/T2V/V2V` -> `<base>` -> `BaseModel/Lora/VAE` -> モデルを選択し `Apply`
   - Flat: 従来どおり系譜ドロップダウンで絞り込み、`Set T2I/I2I/T2V/I2V`
   - `Rescan` でキャッシュを破棄して再走査、`Reveal` でエクスプローラを開く
4. `Text to Image` / `Image to Image` / `Text to Video` / `Image to Video` タブで生成ジョブを開始
5. 下部ステータスで進捗を確認し、完了後に画像または動画をプレビュー

LoRA は各生成タブで複数選択できます。`LoRA Scale` は選択したすべての LoRA に同じ値で適用されます。  
VAE は `Text to Image` / `Image to Image` タブで選択できます。

`Models` タブの `Download Save Path` に保存先パスを入れると、ダウンロードごとに保存先を選択できます。  
未入力の場合は `Settings` の `Models Directory` が使われます。

## Model Search API（拡張）

既存の `GET /api/models/search` は後方互換のまま維持されています。  
新UIは `search2/detail` を利用します。

- `GET /api/models/search2`
  - query:
    - `task`（必須）: `text-to-image|image-to-image|text-to-video|image-to-video`
    - `source`: `all|huggingface|civitai`
    - `query`: 検索語（空欄可）
    - `base_model`: ベースモデルフィルタ
    - `sort`: `popularity|downloads|likes|updated|created`
    - `nsfw`: `include|exclude`
    - `model_kind`: `checkpoint|lora|vae|controlnet|embedding|upscaler`（provider対応範囲内）
    - `limit`: 1-100
    - `page` または `cursor`
  - response:
    - `items[]`
    - `next_cursor`, `prev_cursor`
    - `page_info`

- `GET /api/models/detail`
  - query:
    - `source`: `huggingface|civitai`
    - `id`: HF repo id または CivitAI id（`civitai/123` 形式可）
  - response:
    - `title/name/id/source`
    - `description`
    - `tags[]`
    - `previews[]`
    - `versions[]`（`files[]` 含む）

- `POST /api/models/download`（後方互換拡張）
  - 従来:
    - `repo_id`, `revision`, `target_dir`
  - 追加:
    - `source`（`huggingface|civitai`）
    - `hf_revision`
    - `civitai_model_id`, `civitai_version_id`, `civitai_file_id`
  - 省略時は従来動作
  - ダウンロード後に `model_meta.json` を保存（CivitAIは `civitai_model.json` も保存）
  - `videogen_meta.json` を保存（task/base_model/category/source など）

- `GET /api/tasks`（新規）
  - query:
    - `task_type`（任意）
    - `status`（`all|queued|running|completed|error`）
    - `limit`（1-200）
  - response:
    - `tasks[]`（`id,task_type,status,progress,message,created_at,updated_at,downloaded_bytes,total_bytes,result,error`）

## Local Models Tree API（新規）

- `GET /api/models/local/tree`
  - query:
    - `dir`（任意）: モデルルート。未指定時は `settings.paths.models_dir`
  - response:
    - `model_root`
    - `generated_at`
    - `tasks[]`:
      - `task` (`T2I|I2I|T2V|V2V`)
      - `bases[]` (`base_name`)
      - `categories[]` (`BaseModel|Lora|VAE`)
      - `items[]` (`name,path,size_bytes,provider,task_api,apply_supported,...`)
    - `flat_items[]`（後方互換UI用）

- `POST /api/models/local/rescan`
  - body:
    - `dir`（任意）
  - response:
    - `GET /api/models/local/tree` と同形式（再走査後）

- `POST /api/models/local/reveal`
  - body:
    - `path`（必須）: 開きたいローカルパス
    - `base_dir`（任意）: 安全チェック基準ディレクトリ
  - response:
    - Windows: `{status:\"ok\", path:\"...\"}`
    - 非Windows: `{status:\"not_supported\", reason:\"...\"}`

## Runtime / Task Ops API（追加）

- `GET /api/runtime`
  - ROCm/CUDA 利用可否、dtype 方針、AOTriton/allocator env、GPUメモリ情報を返します。
  - `GET /api/system/info` は後方互換で維持されています。

- `POST /api/tasks/cancel`
  - body:
    - `task_id`（必須）
  - response:
    - `{status: "ok", task: {...}}`
  - 備考:
    - 生成系・CivitAIストリームDLはキャンセル反映されます。
    - HF `snapshot_download` は処理開始後に即中断できない場合があります（サーバログに明示）。

- `POST /api/cleanup`
  - body:
    - `include_cache`（任意, 既定 `true`）
  - response:
    - 削除件数・削除パス・キャッシュ前後サイズ
  - 備考:
    - `settings.storage.*` の上限/TTLポリシーを使って `outputs/tmp/HF cache` を整理します。

## 手動テスト手順（Model Search）

1. `start.bat` で起動し、`Models` タブを開く
2. 検索条件を設定
   - `Source=all`, `Task=text-to-image`, `Limit=30`
   - 必要に応じて `Sort/NSFW/Model Kind/Base Model` を設定
3. `Search Models` を実行
   - カードが Grid/List で表示されること
   - `Prev/Next` でページ遷移できること
4. 任意カードの `Detail` を押す
   - 右ペインに説明、タグ、プレビュー、バージョン/ファイルが表示されること
5. ダウンロード
   - HF: revision（select/manual）を指定して `Download`
   - CivitAI: version/file を選択して `Download`
6. 完了後、`Local Models` を更新
   - ダウンロード済みモデルが表示されること
7. `Apply` を押す
   - 選択中 task のモデル選択へ反映されること

## 手動テスト手順（Local Models Tree）

1. `models` 配下に次のようにダミーファイルを配置
   - `models/T2I/SDXL1.0/BaseModel/demo.safetensors`
   - `models/T2I/SDXL1.0/Lora/style.safetensors`
   - `models/T2I/SDXL1.0/VEA/vae.safetensors`（表示は `VAE` に正規化）
2. `Local Models` タブで `Refresh Local List` を押下
3. `View Mode=Tree` で `T2I -> SDXL1.0 -> BaseModel/Lora/VAE` が見えることを確認
4. `Tree Search` に `demo` を入力し、該当枝だけ表示されることを確認
5. `demo.safetensors` を選択し `Apply` で T2I の Model Selection に反映されることを確認
6. `Rescan` を押下し、追加したモデルが反映されることを確認
7. `Reveal` を押下し Explorer で対象フォルダ/ファイルが開くことを確認（Windows）

## E2E スモーク手順（Windows + ROCm）

1. `start.bat` で起動
2. `GET /api/runtime` を確認し、`device` / `dtype` / `torch_hip_version` / `hardware_profile` / `load_policy_preview` が期待通りであることを確認
3. `Models` タブで検索し、`Detail` を開いてローディング表示が出ることを確認
4. 任意モデルをダウンロード開始し、右上 Downloads ウィジェットで進捗表示を確認
5. `Text to Video` または `Image to Video` を実行し、進捗バーと `Task Log` の更新を確認
6. 実行中に `Cancel Task` を押して `cancelled` 遷移を確認
7. `Settings` で `Run Cleanup` を押し、削除件数メッセージが表示されることを確認

## 保存先

- 設定: `data/settings.json`
  - `server.rocm_aotriton_experimental`: `true/false`（`start.bat` 起動時に `TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1/0` へ反映）
  - `server.preferred_dtype`: `bf16|float16`（既定: `bf16`, 非対応時は `float16` に自動フォールバック）
  - `server.vram_gpu_direct_load_threshold_gb`: VRAM直ロード判定しきい値（既定 `48`）
  - `server.enable_device_map_auto`: 低VRAM時の `device_map=\"auto\"` を有効化（既定 `true`）
  - `server.enable_model_cpu_offload`: 低VRAM時の CPU offload フォールバックを有効化（既定 `true`）
  - `server.gpu_max_concurrency`: 生成系の同時実行上限（既定 `1`）
  - `server.allow_software_video_fallback`: `true` の場合、AMF失敗時に `libx264` へフォールバック
  - `server.request_timeout_sec` / `server.request_retry_count` / `server.request_retry_backoff_sec`
  - `storage.cleanup_enabled`
  - `storage.cleanup_max_age_days`
  - `storage.cleanup_max_outputs_count`
  - `storage.cleanup_max_tmp_count`
  - `storage.cleanup_max_cache_size_gb`
- モデル: `models/`
- 生成画像/動画: `outputs/`
- 一時画像: `tmp/`
- ログ: `logs/YYYYMMDD_HHMMSS_videogen_pid<process-id>.log`（起動プロセスごとに作成、`data/settings.json` の `paths.logs_dir` で変更可）

## 開発ツール

- `ruff` / `black` / `pytest` を必須化
- 開発依存:
  - `requirements-test.txt`（テスト + lint）
  - `requirements-dev.txt`（最小開発セット）
- 実行:
  - `python -m compileall .`
  - `python -m black main.py videogen tests`
  - `python -m ruff check .`
  - `python -m pytest -q`
- 設計判断の詳細: `docs/ARCHITECTURE.md`

## ログ確認

- 直近ログAPI: `GET /api/logs/recent?lines=200`
- 失敗時はタスク詳細の `error` と、ログの `Traceback` / `diagnostics` を確認してください。

## 注意

- 初回モデル読み込みは時間がかかります。
- VRAM 使用量が大きいため、`duration_seconds` や `steps` を調整してください。
- 入力モデルIDは Hugging Face repo ID またはローカルパスを使用できます。

## FramePack チューニング

- `VIDEOGEN_FRAMEPACK_SEGMENT_FRAMES`（既定: `16`）
  - 1パックあたりの生成フレーム数（小さいほどVRAMピークを抑制）
- `VIDEOGEN_FRAMEPACK_OVERLAP_FRAMES`（既定: `2`）
  - パック間の重なりフレーム数（接続滑らかさ用）
- `VIDEOGEN_FRAMEPACK_LONG_SEGMENT_FRAMES`（既定: `8`）
  - 長尺（30分以上）時に適用する最大パック長
- 後方互換として `VIDEOGEN_VIDEO_CHUNK_FRAMES` も引き続き利用可能です。
