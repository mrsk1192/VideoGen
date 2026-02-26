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
3. `Local Models` タブで系譜ドロップダウンで絞り込み、`Set T2I/I2I/T2V/I2V` を使って生成タブへ反映
4. `Text to Image` / `Image to Image` / `Text to Video` / `Image to Video` タブで生成ジョブを開始
5. 下部ステータスで進捗を確認し、完了後に画像または動画をプレビュー

LoRA は各生成タブで複数選択できます。`LoRA Scale` は選択したすべての LoRA に同じ値で適用されます。  
VAE は `Text to Image` / `Image to Image` タブで選択できます。

`Models` タブの `Download Save Path` に保存先パスを入れると、ダウンロードごとに保存先を選択できます。  
未入力の場合は `Settings` の `Models Directory` が使われます。

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
