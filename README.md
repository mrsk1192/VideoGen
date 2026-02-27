# ROCm VideoGen Web App

ROCm 環境で動作する `Text-to-Image` / `Image-to-Image` / `Text-to-Video` / `Image-to-Video` の Web アプリです。

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

## 保存先

- 設定: `data/settings.json`
  - `server.rocm_aotriton_experimental`: `true/false`（`start.bat` 起動時に `TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1/0` へ反映）
- モデル: `models/`
- 生成画像/動画: `outputs/`
- 一時画像: `tmp/`
- ログ: `logs/YYYYMMDD_HHMMSS_videogen_pid<process-id>.log`（起動プロセスごとに作成、`data/settings.json` の `paths.logs_dir` で変更可）

## ログ確認

- 直近ログAPI: `GET /api/logs/recent?lines=200`
- 失敗時はタスク詳細の `error` と、ログの `Traceback` / `diagnostics` を確認してください。

## 注意

- 初回モデル読み込みは時間がかかります。
- VRAM 使用量が大きいため、`frames` や `steps` を調整してください。
- 入力モデルIDは Hugging Face repo ID またはローカルパスを使用できます。
