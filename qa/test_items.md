# 試験項目

| ID | レベル | 試験項目 | 期待結果 | 自動試験 |
|---|---|---|---|---|
| U-01 | Unit | `sanitize_repo_id` / `desanitize_repo_id` 往復 | 元のrepo_idに戻る | `tests/unit/test_helpers.py::test_sanitize_desanitize_roundtrip` |
| U-02 | Unit | `deep_merge` のネストマージ | ネスト値が上書きされ、非変更値は保持 | `tests/unit/test_helpers.py::test_deep_merge_nested` |
| U-03 | Unit | `parse_civitai_model_id` の判定 | `civitai/<id>` 形式のみ数値IDを返す | `tests/unit/test_helpers.py::test_parse_civitai_model_id` |
| U-04 | Unit | `find_local_preview_relpath` の優先検出 | 期待するサムネイル相対パスを返す | `tests/unit/test_helpers.py::test_find_local_preview_relpath` |
| I-01 | Integration | `GET/PUT /api/settings` | 保存後に同値を取得できる | `tests/integration/test_api.py::test_settings_roundtrip` |
| I-02 | Integration | `GET /api/models/local?dir=...` | 指定ディレクトリ配下モデルを列挙 | `tests/integration/test_api.py::test_models_local_with_custom_dir` |
| I-03 | Integration | `POST /api/models/download` (Hugging Face) | タスクが `queued/running -> completed` 遷移 | `tests/integration/test_api.py::test_download_task_lifecycle` |
| I-04 | Integration | `POST /api/models/download` (CivitAI) | CivitAIモデルを取得してローカル保存できる | `tests/integration/test_api.py::test_download_task_lifecycle_civitai` |
| I-05 | Integration | `GET /api/models/search` の検索元フィルタ | `source` 指定でHuggingFace/CivitAIを切替 | `tests/integration/test_api.py::test_models_search_source_filter` |
| I-06 | Integration | `GET /api/models/catalog` | `items` と `default_model` が返る | `tests/integration/test_api.py::test_model_catalog_endpoint` |
| S-01 | System | 画面初期表示 | ページが表示され主要要素が操作可能 | `tests/system/test_ui_e2e.py::test_local_models_filter_and_model_thumbnail` |
| S-02 | System | ローカル系譜フィルタ選択 | ローカルモデル画面でドロップダウン操作が可能 | `tests/system/test_ui_e2e.py::test_local_models_filter_and_model_thumbnail` |
| S-03 | System | モデル選択変更 | サムネイルが表示される | `tests/system/test_ui_e2e.py::test_local_models_filter_and_model_thumbnail` |
