# ROCm VideoGen Web App

ROCm 環境で動作する `Text-to-Video` / `Image-to-Video` の Web アプリです。

## 機能

- Web GUI からテキスト→動画生成
- Web GUI から画像→動画生成
- Hugging Face モデル検索
- 検索語なしでもモデル検索可能（タスク別の人気順）
- 検索結果にプレビュー画像がある場合は表示
- 生成モデルはドロップダウンから選択可能
- モデル選択時にサムネイル表示
- サーバサイドでモデルをダウンロード
- 設定値保存 (`data/settings.json`)
- 生成ジョブの進捗表示と動画プレビュー
- Web GUI 多言語表示（`en`, `ja`, `es`, `fr`, `de`, `it`, `pt`, `ru`, `ar`）
- モデル保存先をフルパスで選択可能（フォルダブラウザ対応）

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

ブラウザで `http://localhost:8000` を開いて利用します。

## 使い方

1. `Settings` タブで保存先ディレクトリやデフォルトモデルを確認・保存
2. `Models` タブでモデルを検索し、`Download` を実行（サーバ側に保存）
3. `Text to Video` または `Image to Video` タブで生成ジョブを開始
4. 下部ステータスで進捗を確認し、完了後に動画をプレビュー

`Models` タブの `Download Save Path` に保存先パスを入れると、ダウンロードごとに保存先を選択できます。  
未入力の場合は `Settings` の `Models Directory` が使われます。

## 保存先

- 設定: `data/settings.json`
- モデル: `models/`
- 生成動画: `outputs/`
- 一時画像: `tmp/`

## 注意

- 初回モデル読み込みは時間がかかります。
- VRAM 使用量が大きいため、`frames` や `steps` を調整してください。
- 入力モデルIDは Hugging Face repo ID またはローカルパスを使用できます。
