from pathlib import Path

import json
from PIL import Image

import main


BASE = Path(__file__).resolve().parents[2]
RUNTIME = BASE / "artifacts" / "tests" / "system_runtime"
MODELS = RUNTIME / "models"
OUTPUTS = RUNTIME / "outputs"
TMP = RUNTIME / "tmp"


def _prepare_runtime() -> None:
    MODELS.mkdir(parents=True, exist_ok=True)
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    TMP.mkdir(parents=True, exist_ok=True)
    settings_path = RUNTIME / "settings.json"

    model_dir = MODELS / "example-org--example-model"
    model_dir.mkdir(parents=True, exist_ok=True)
    model_index = model_dir / "model_index.json"
    model_index.write_text(json.dumps({"_class_name": "TextToVideoSDPipeline"}), encoding="utf-8")
    thumb = model_dir / "thumbnail.png"
    if not thumb.exists():
        img = Image.new("RGB", (320, 180), color=(20, 80, 120))
        img.save(thumb)

    # Keep test runtime settings isolated from user settings.
    main.settings_store = main.SettingsStore(settings_path, main.DEFAULT_SETTINGS)
    main.settings_store.update(
        {
            "paths": {
                "models_dir": str(MODELS),
                "outputs_dir": str(OUTPUTS),
                "tmp_dir": str(TMP),
            },
            "defaults": {
                "text2video_model": "example-org/example-model",
                "image2video_model": "example-org/example-model",
            },
        }
    )
    main.ensure_runtime_dirs(main.settings_store.get())


def _disable_remote_catalog() -> None:
    def fake_search_hf_models(task, query, limit, token):
        return []

    def fake_search_civitai_models(task, query, limit):
        return []

    main.search_hf_models = fake_search_hf_models
    main.search_civitai_models = fake_search_civitai_models


_prepare_runtime()
_disable_remote_catalog()

app = main.app
