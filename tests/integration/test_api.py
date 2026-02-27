import json
import os
import time
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

import main

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def client() -> TestClient:
    return TestClient(main.app)


@pytest.fixture(autouse=True, scope="module")
def isolate_settings_store(tmp_path_factory: pytest.TempPathFactory):
    original_store = main.settings_store
    settings_path = tmp_path_factory.mktemp("settings") / "settings.json"
    main.settings_store = main.SettingsStore(settings_path, main.DEFAULT_SETTINGS)
    main.ensure_runtime_dirs(main.settings_store.get())
    yield
    main.settings_store = original_store
    main.ensure_runtime_dirs(main.settings_store.get())


def test_settings_roundtrip(client: TestClient) -> None:
    got = client.get("/api/settings")
    assert got.status_code == 200
    payload = got.json()
    payload["paths"]["tmp_dir"] = "tmp"
    payload["defaults"]["fps"] = 12

    updated = client.put("/api/settings", json=payload)
    assert updated.status_code == 200
    assert updated.json()["defaults"]["fps"] == 12

    verify = client.get("/api/settings")
    assert verify.status_code == 200
    assert verify.json()["defaults"]["fps"] == 12


def test_models_local_with_custom_dir(client: TestClient, tmp_path: Path) -> None:
    custom = tmp_path / "my-models"
    (custom / "org--model").mkdir(parents=True)
    resp = client.get("/api/models/local", params={"dir": str(custom)})
    assert resp.status_code == 200
    body = resp.json()
    assert body["base_dir"] == str(custom.resolve())
    assert any(item["name"] == "org--model" for item in body["items"])


def test_delete_local_model_endpoint(client: TestClient, tmp_path: Path) -> None:
    base = tmp_path / "delete-target"
    model_dir = base / "org--sample"
    (model_dir / ".cache" / "huggingface").mkdir(parents=True)
    (model_dir / "weights.safetensors").write_bytes(b"123")

    listing = client.get("/api/models/local", params={"dir": str(base)})
    assert listing.status_code == 200
    item = next(x for x in listing.json()["items"] if x["name"] == "org--sample")
    assert item["can_delete"] is True

    resp = client.post("/api/models/local/delete", json={"model_name": "org--sample", "base_dir": str(base)})
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"
    assert not model_dir.exists()


def test_delete_local_model_rejects_non_model_directory(client: TestClient, tmp_path: Path) -> None:
    base = tmp_path / "delete-invalid"
    target = base / "random-dir"
    target.mkdir(parents=True)
    (target / "note.txt").write_text("hello", encoding="utf-8")

    resp = client.post("/api/models/local/delete", json={"model_name": "random-dir", "base_dir": str(base)})
    assert resp.status_code == 400


def test_delete_local_model_by_path_nested_directory(client: TestClient, tmp_path: Path) -> None:
    base = tmp_path / "delete-by-path-nested"
    model_dir = base / "T2I" / "SDXL1.0" / "BaseModel" / "nested-model"
    model_dir.mkdir(parents=True)
    (model_dir / "model_index.json").write_text(json.dumps({"_class_name": "StableDiffusionXLPipeline"}), encoding="utf-8")
    (model_dir / "model.safetensors").write_bytes(b"abc")

    tree_resp = client.get("/api/models/local/tree", params={"dir": str(base)})
    assert tree_resp.status_code == 200
    flat_items = tree_resp.json().get("flat_items", [])
    nested_item = next(item for item in flat_items if Path(item["path"]).name == "nested-model")
    assert nested_item["can_delete"] is True

    resp = client.post(
        "/api/models/local/delete",
        json={"path": nested_item["path"], "base_dir": str(base)},
    )
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"
    assert not model_dir.exists()


def test_delete_local_model_by_path_file(client: TestClient, tmp_path: Path) -> None:
    base = tmp_path / "delete-by-path-file"
    model_file = base / "T2I" / "SD1.5" / "BaseModel" / "single-model.safetensors"
    model_file.parent.mkdir(parents=True, exist_ok=True)
    model_file.write_bytes(b"abc")

    resp = client.post(
        "/api/models/local/delete",
        json={"path": str(model_file.relative_to(base)), "base_dir": str(base)},
    )
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"
    assert not model_file.exists()


def test_outputs_list_and_delete(client: TestClient, tmp_path: Path) -> None:
    outputs_dir = tmp_path / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (64, 64), color=(1, 2, 3)).save(outputs_dir / "sample.png")
    (outputs_dir / "sample.mp4").write_bytes(b"\x00\x00\x00\x18ftypmp42")

    settings = client.get("/api/settings").json()
    settings["paths"]["outputs_dir"] = str(outputs_dir.resolve())
    put_resp = client.put("/api/settings", json=settings)
    assert put_resp.status_code == 200

    listing = client.get("/api/outputs", params={"limit": 50})
    assert listing.status_code == 200
    body = listing.json()
    assert body["base_dir"] == str(outputs_dir.resolve())
    items = body["items"]
    names = {item["name"] for item in items}
    assert "sample.png" in names
    assert "sample.mp4" in names
    image_item = next(item for item in items if item["name"] == "sample.png")
    assert image_item["kind"] == "image"
    assert image_item["view_url"].startswith("/api/images/")
    video_item = next(item for item in items if item["name"] == "sample.mp4")
    assert video_item["kind"] == "video"
    assert video_item["view_url"].startswith("/api/videos/")

    delete_resp = client.post("/api/outputs/delete", json={"file_name": "sample.png"})
    assert delete_resp.status_code == 200
    assert not (outputs_dir / "sample.png").exists()


def test_outputs_delete_rejects_invalid_name(client: TestClient) -> None:
    resp = client.post("/api/outputs/delete", json={"file_name": "../escape.txt"})
    assert resp.status_code == 400


def test_download_task_lifecycle(client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_snapshot_download(repo_id: str, local_dir: str, **kwargs):
        path = Path(local_dir)
        path.mkdir(parents=True, exist_ok=True)
        (path / "model_index.json").write_text(json.dumps({"repo_id": repo_id}), encoding="utf-8")
        return str(path)

    monkeypatch.setattr(main, "snapshot_download", fake_snapshot_download)
    target_dir = tmp_path / "download-target"
    post = client.post("/api/models/download", json={"repo_id": "foo/bar", "target_dir": str(target_dir)})
    assert post.status_code == 200
    task_id = post.json()["task_id"]

    deadline = time.time() + 5
    last = None
    while time.time() < deadline:
        status = client.get(f"/api/tasks/{task_id}")
        assert status.status_code == 200
        last = status.json()
        if last["status"] in ("completed", "error"):
            break
        time.sleep(0.1)

    assert last is not None
    assert last["status"] == "completed"
    assert Path(last["result"]["local_path"]).exists()


def test_download_task_lifecycle_civitai(client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeResponse:
        def __init__(self, payload: bytes, content_length: int | None = None) -> None:
            self._payload = payload
            self._offset = 0
            self.headers = {"Content-Length": str(content_length)} if content_length is not None else {}

        def read(self, size: int = -1) -> bytes:
            if size is None or size < 0:
                data = self._payload[self._offset :]
                self._offset = len(self._payload)
                return data
            if self._offset >= len(self._payload):
                return b""
            end = min(self._offset + size, len(self._payload))
            data = self._payload[self._offset : end]
            self._offset = end
            return data

        def close(self) -> None:
            return None

    model_payload = {
        "id": 123,
        "name": "Civit Model",
        "modelVersions": [
            {
                "id": 456,
                "name": "v1",
                "files": [
                    {
                        "id": 789,
                        "name": "model.safetensors",
                        "type": "Model",
                        "downloadUrl": "https://example.invalid/model.safetensors",
                    }
                ],
            }
        ],
    }
    model_bytes = b"abcdef"

    def fake_urlopen(request_obj, timeout: int = 20):
        url = request_obj.full_url
        if "/api/v1/models/123" in url:
            return FakeResponse(json.dumps(model_payload).encode("utf-8"))
        if "example.invalid/model.safetensors" in url:
            return FakeResponse(model_bytes, content_length=len(model_bytes))
        raise AssertionError(f"unexpected url: {url}")

    monkeypatch.setattr(main, "urlopen", fake_urlopen)

    target_dir = tmp_path / "download-target-civitai"
    post = client.post("/api/models/download", json={"repo_id": "civitai/123", "target_dir": str(target_dir)})
    assert post.status_code == 200
    task_id = post.json()["task_id"]

    deadline = time.time() + 5
    last = None
    while time.time() < deadline:
        status = client.get(f"/api/tasks/{task_id}")
        assert status.status_code == 200
        last = status.json()
        if last["status"] in ("completed", "error"):
            break
        time.sleep(0.1)

    assert last is not None
    assert last["status"] == "completed"
    model_dir = Path(last["result"]["local_path"])
    assert model_dir.exists()
    assert (model_dir / "model.safetensors").read_bytes() == model_bytes
    metadata = json.loads((model_dir / "civitai_model.json").read_text(encoding="utf-8"))
    assert metadata["model_id"] == 123


def test_model_catalog_endpoint(client: TestClient) -> None:
    resp = client.get("/api/models/catalog", params={"task": "text-to-video", "limit": 5})
    assert resp.status_code == 200
    body = resp.json()
    assert "items" in body
    assert "default_model" in body


def test_models_search_merges_hf_and_civitai(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        main,
        "search_hf_models",
        lambda task, query, limit, token: [
            {
                "id": "hf/a",
                "pipeline_tag": task,
                "downloads": 100,
                "likes": 10,
                "size_bytes": 1,
                "source": "huggingface",
                "download_supported": True,
            }
        ],
    )
    monkeypatch.setattr(
        main,
        "search_civitai_models",
        lambda task, query, limit: [
            {
                "id": "civitai/1",
                "pipeline_tag": task,
                "downloads": 90,
                "likes": 9,
                "size_bytes": 2,
                "source": "civitai",
                "download_supported": True,
            }
        ],
    )
    resp = client.get("/api/models/search", params={"task": "text-to-image", "query": "test", "limit": 6})
    assert resp.status_code == 200
    ids = [item["id"] for item in resp.json()["items"]]
    assert "hf/a" in ids
    assert "civitai/1" in ids


def test_models_search2_returns_page_info_and_installed(client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    models_dir = tmp_path / "models_search2"
    (models_dir / "hf--a").mkdir(parents=True)
    settings = client.get("/api/settings").json()
    settings["paths"]["models_dir"] = str(models_dir.resolve())
    put_resp = client.put("/api/settings", json=settings)
    assert put_resp.status_code == 200

    monkeypatch.setattr(
        main,
        "search_hf_models_v2",
        lambda **_kwargs: {
            "items": [
                {
                    "id": "hf/a",
                    "title": "hf/a",
                    "pipeline_tag": "text-to-image",
                    "downloads": 1,
                    "likes": 2,
                    "size_bytes": 3,
                    "source": "huggingface",
                    "download_supported": True,
                    "base_model": "StableDiffusion XL",
                }
            ],
            "has_next": True,
        },
    )
    monkeypatch.setattr(main, "search_civitai_models_v2", lambda **_kwargs: {"items": [], "has_next": False})

    resp = client.get(
        "/api/models/search2",
        params={"task": "text-to-image", "query": "", "limit": 30, "source": "huggingface", "page": 1},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["page_info"]["page"] == 1
    assert body["next_cursor"] == "2"
    assert body["items"][0]["installed"] is True


def test_models_search2_filters_by_size_mb(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        main,
        "search_hf_models_v2",
        lambda **_kwargs: {
            "items": [
                {
                    "id": "hf/small",
                    "title": "hf/small",
                    "pipeline_tag": "text-to-image",
                    "size_bytes": 400 * 1024 * 1024,
                    "source": "huggingface",
                },
                {
                    "id": "hf/mid",
                    "title": "hf/mid",
                    "pipeline_tag": "text-to-image",
                    "size_bytes": 1500 * 1024 * 1024,
                    "source": "huggingface",
                },
                {"id": "hf/unknown", "title": "hf/unknown", "pipeline_tag": "text-to-image", "size_bytes": None, "source": "huggingface"},
            ],
            "has_next": False,
        },
    )
    monkeypatch.setattr(
        main,
        "search_civitai_models_v2",
        lambda **_kwargs: {
            "items": [
                {
                    "id": "civitai/large",
                    "title": "civitai/large",
                    "pipeline_tag": "text-to-image",
                    "size_bytes": 2500 * 1024 * 1024,
                    "source": "civitai",
                },
                {
                    "id": "civitai/huge",
                    "title": "civitai/huge",
                    "pipeline_tag": "text-to-image",
                    "size_bytes": 5000 * 1024 * 1024,
                    "source": "civitai",
                },
            ],
            "has_next": False,
        },
    )

    resp = client.get(
        "/api/models/search2",
        params={
            "task": "text-to-image",
            "query": "",
            "source": "all",
            "size_min_mb": 1000,
            "size_max_mb": 3000,
        },
    )
    assert resp.status_code == 200
    ids = [item["id"] for item in resp.json()["items"]]
    assert "hf/mid" in ids
    assert "civitai/large" in ids
    assert "hf/small" not in ids
    assert "hf/unknown" not in ids
    assert "civitai/huge" not in ids


def test_models_search2_rejects_invalid_size_range(client: TestClient) -> None:
    resp = client.get(
        "/api/models/search2",
        params={"task": "text-to-image", "query": "", "size_min_mb": 3000, "size_max_mb": 1000},
    )
    assert resp.status_code == 400
    assert "size_max_mb" in str(resp.json().get("detail") or "")


def test_models_detail_endpoints(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        main,
        "get_hf_model_detail",
        lambda repo_id, token: {"source": "huggingface", "id": repo_id, "title": repo_id, "versions": [], "previews": [], "tags": []},
    )
    monkeypatch.setattr(
        main,
        "get_civitai_model_detail",
        lambda model_id: {"source": "civitai", "id": f"civitai/{model_id}", "title": "x", "versions": [], "previews": [], "tags": []},
    )

    hf = client.get("/api/models/detail", params={"source": "huggingface", "id": "foo/bar"})
    assert hf.status_code == 200
    assert hf.json()["id"] == "foo/bar"

    civitai = client.get("/api/models/detail", params={"source": "civitai", "id": "civitai/123"})
    assert civitai.status_code == 200
    assert civitai.json()["id"] == "civitai/123"


def test_models_search_source_filter(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    called = {"hf": 0, "civitai": 0}

    def fake_hf(task, query, limit, token):
        called["hf"] += 1
        return [
            {
                "id": "hf/only",
                "pipeline_tag": task,
                "downloads": 10,
                "likes": 1,
                "size_bytes": 1,
                "source": "huggingface",
                "download_supported": True,
            }
        ]

    def fake_civitai(task, query, limit):
        called["civitai"] += 1
        return [
            {
                "id": "civitai/only",
                "pipeline_tag": task,
                "downloads": 9,
                "likes": 1,
                "size_bytes": 1,
                "source": "civitai",
                "download_supported": True,
            }
        ]

    monkeypatch.setattr(main, "search_hf_models", fake_hf)
    monkeypatch.setattr(main, "search_civitai_models", fake_civitai)

    hf_resp = client.get("/api/models/search", params={"task": "text-to-image", "query": "", "limit": 6, "source": "huggingface"})
    assert hf_resp.status_code == 200
    hf_ids = [item["id"] for item in hf_resp.json()["items"]]
    assert hf_ids == ["hf/only"]

    civitai_resp = client.get("/api/models/search", params={"task": "text-to-image", "query": "", "limit": 6, "source": "civitai"})
    assert civitai_resp.status_code == 200
    civitai_ids = [item["id"] for item in civitai_resp.json()["items"]]
    assert civitai_ids == ["civitai/only"]

    assert called["hf"] == 1
    assert called["civitai"] == 1


def test_models_search_civitai_live_parser_path(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(main, "search_hf_models", lambda task, query, limit, token: [])

    class FakeResponse:
        def __init__(self, payload: dict) -> None:
            self._payload = payload

        def read(self) -> bytes:
            return json.dumps(self._payload).encode("utf-8")

        def close(self) -> None:
            return None

    payload = {
        "items": [
            {
                "id": 12345,
                "name": "Example CivitAI",
                "type": "Checkpoint",
                "stats": {"downloadCount": 1200, "favoriteCount": 12},
                "modelVersions": [
                    {
                        "images": [{"url": "https://example.invalid/preview.jpg"}],
                        "files": [{"sizeKB": 2048, "downloadUrl": "https://example.invalid/model.safetensors"}],
                    }
                ],
            }
        ]
    }

    def fake_urlopen(request_obj, timeout: int = 20):
        assert "civitai.com/api/v1/models" in request_obj.full_url
        assert timeout == 20
        return FakeResponse(payload)

    monkeypatch.setattr(main, "urlopen", fake_urlopen)

    resp = client.get("/api/models/search", params={"task": "text-to-image", "query": "", "limit": 6})
    assert resp.status_code == 200
    body = resp.json()
    civitai = next(item for item in body["items"] if item["id"] == "civitai/12345")
    assert civitai["source"] == "civitai"
    assert civitai["download_supported"] is True
    assert civitai["preview_url"] == "https://example.invalid/preview.jpg"
    assert civitai["size_bytes"] == 2048 * 1024


def test_download_task_lifecycle_civitai_with_selected_file(client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeResponse:
        def __init__(self, payload: bytes, content_length: int | None = None) -> None:
            self._payload = payload
            self._offset = 0
            self.headers = {"Content-Length": str(content_length)} if content_length is not None else {}

        def read(self, size: int = -1) -> bytes:
            if size is None or size < 0:
                data = self._payload[self._offset :]
                self._offset = len(self._payload)
                return data
            if self._offset >= len(self._payload):
                return b""
            end = min(self._offset + size, len(self._payload))
            data = self._payload[self._offset : end]
            self._offset = end
            return data

        def close(self) -> None:
            return None

    model_payload = {
        "id": 123,
        "name": "Civit Model",
        "modelVersions": [
            {
                "id": 456,
                "name": "v1",
                "images": [{"url": "https://example.invalid/preview.jpg"}],
                "files": [{"id": 789, "name": "file_a.safetensors", "type": "Model", "downloadUrl": "https://example.invalid/file_a"}],
            },
            {
                "id": 999,
                "name": "v2",
                "files": [{"id": 555, "name": "file_b.safetensors", "type": "Model", "downloadUrl": "https://example.invalid/file_b"}],
            },
        ],
    }

    def fake_urlopen(request_obj, timeout: int = 20):
        url = request_obj.full_url
        if "/api/v1/models/123" in url:
            return FakeResponse(json.dumps(model_payload).encode("utf-8"))
        if "example.invalid/file_b" in url:
            payload = b"selected"
            return FakeResponse(payload, content_length=len(payload))
        if "example.invalid/preview.jpg" in url:
            payload = b"img"
            return FakeResponse(payload, content_length=len(payload))
        raise AssertionError(f"unexpected url: {url}")

    monkeypatch.setattr(main, "urlopen", fake_urlopen)
    target_dir = tmp_path / "download-target-civitai-selected"
    post = client.post(
        "/api/models/download",
        json={
            "repo_id": "civitai/123",
            "source": "civitai",
            "civitai_model_id": 123,
            "civitai_version_id": 999,
            "civitai_file_id": 555,
            "target_dir": str(target_dir),
        },
    )
    assert post.status_code == 200
    task_id = post.json()["task_id"]

    deadline = time.time() + 5
    last = None
    while time.time() < deadline:
        status = client.get(f"/api/tasks/{task_id}")
        assert status.status_code == 200
        last = status.json()
        if last["status"] in ("completed", "error"):
            break
        time.sleep(0.1)

    assert last is not None
    assert last["status"] == "completed"
    model_dir = Path(last["result"]["local_path"])
    assert (model_dir / "file_b.safetensors").read_bytes() == b"selected"


def test_model_catalog_filters_incompatible_local_models(client: TestClient, tmp_path: Path) -> None:
    models_dir = tmp_path / "models"
    t2i = models_dir / "ok--text-image"
    i2i = models_dir / "ok--image-image"
    t2v = models_dir / "ok--text"
    i2v = models_dir / "ok--image"
    lora = models_dir / "bad--lora"
    t2i.mkdir(parents=True)
    i2i.mkdir(parents=True)
    t2v.mkdir(parents=True)
    i2v.mkdir(parents=True)
    lora.mkdir(parents=True)
    (t2i / "model_index.json").write_text(json.dumps({"_class_name": "StableDiffusionPipeline"}), encoding="utf-8")
    (i2i / "model_index.json").write_text(json.dumps({"_class_name": "StableDiffusionImg2ImgPipeline"}), encoding="utf-8")
    (t2v / "model_index.json").write_text(json.dumps({"_class_name": "TextToVideoSDPipeline"}), encoding="utf-8")
    (i2v / "model_index.json").write_text(json.dumps({"_class_name": "I2VGenXLPipeline"}), encoding="utf-8")
    (lora / "README.md").write_text("no pipeline", encoding="utf-8")

    settings = client.get("/api/settings").json()
    settings["paths"]["models_dir"] = str(models_dir.resolve())
    put_resp = client.put("/api/settings", json=settings)
    assert put_resp.status_code == 200

    t2i_resp = client.get("/api/models/catalog", params={"task": "text-to-image", "limit": 20})
    assert t2i_resp.status_code == 200
    t2i_ids = {item["id"] for item in t2i_resp.json()["items"]}
    assert "ok/text-image" in t2i_ids
    assert "ok/image-image" not in t2i_ids
    assert "ok/text" not in t2i_ids
    assert "ok/image" not in t2i_ids
    assert "bad/lora" not in t2i_ids

    i2i_resp = client.get("/api/models/catalog", params={"task": "image-to-image", "limit": 20})
    assert i2i_resp.status_code == 200
    i2i_ids = {item["id"] for item in i2i_resp.json()["items"]}
    assert "ok/text-image" in i2i_ids
    assert "ok/image-image" in i2i_ids
    assert "ok/text" not in i2i_ids
    assert "ok/image" not in i2i_ids
    assert "bad/lora" not in i2i_ids

    t2v_resp = client.get("/api/models/catalog", params={"task": "text-to-video", "limit": 20})
    assert t2v_resp.status_code == 200
    t2v_ids = {item["id"] for item in t2v_resp.json()["items"]}
    assert "ok/text" in t2v_ids
    assert "ok/image" not in t2v_ids
    assert "bad/lora" not in t2v_ids

    i2v_resp = client.get("/api/models/catalog", params={"task": "image-to-video", "limit": 20})
    assert i2v_resp.status_code == 200
    i2v_ids = {item["id"] for item in i2v_resp.json()["items"]}
    assert "ok/image" in i2v_ids
    assert "ok/text" not in i2v_ids
    assert "bad/lora" not in i2v_ids


def test_local_models_include_preview_and_meta(client: TestClient, tmp_path: Path) -> None:
    models_dir = tmp_path / "models_local_meta"
    model_dir = models_dir / "org--base-model"
    model_dir.mkdir(parents=True)
    (model_dir / "model_index.json").write_text(
        json.dumps({"_class_name": "StableDiffusionPipeline", "_name_or_path": "runwayml/stable-diffusion-v1-5"}),
        encoding="utf-8",
    )
    Image.new("RGB", (32, 32), color=(1, 2, 3)).save(model_dir / "thumbnail.png")
    lora_dir = models_dir / "org--style-lora"
    lora_dir.mkdir(parents=True)
    (lora_dir / "adapter_config.json").write_text(
        json.dumps({"base_model_name_or_path": "runwayml/stable-diffusion-v1-5"}),
        encoding="utf-8",
    )
    (lora_dir / "pytorch_lora_weights.safetensors").write_bytes(b"123")

    resp = client.get("/api/models/local", params={"dir": str(models_dir)})
    assert resp.status_code == 200
    items = resp.json()["items"]
    base_item = next(item for item in items if item["name"] == "org--base-model")
    assert base_item["preview_url"]
    assert "text-to-image" in base_item["compatible_tasks"]
    assert base_item["base_model"] == "runwayml/stable-diffusion-v1-5"
    lora_item = next(item for item in items if item["name"] == "org--style-lora")
    assert lora_item["is_lora"] is True
    assert lora_item["base_model"] == "runwayml/stable-diffusion-v1-5"


def test_local_models_tree_structure_and_vea_alias(client: TestClient, tmp_path: Path) -> None:
    models_dir = tmp_path / "models_tree"
    t2i_base = models_dir / "T2I" / "SDXL1.0"
    (t2i_base / "BaseModel").mkdir(parents=True)
    (t2i_base / "BaseModel" / "realvisxl.safetensors").write_bytes(b"abc")
    (t2i_base / "VEA").mkdir(parents=True)
    (t2i_base / "VEA" / "sdxl_vae.safetensors").write_bytes(b"123")
    i2v_model = models_dir / "I2V" / "Wan2.2" / "BaseModel" / "wan-model"
    i2v_model.mkdir(parents=True)
    (i2v_model / "model_index.json").write_text(json.dumps({"_class_name": "WanImageToVideoPipeline"}), encoding="utf-8")

    resp = client.get("/api/models/local/tree", params={"dir": str(models_dir)})
    assert resp.status_code == 200
    body = resp.json()
    assert body["model_root"] == str(models_dir.resolve())
    tasks = {task["task"]: task for task in body["tasks"]}
    assert "T2I" in tasks
    assert "V2V" in tasks

    t2i_categories = tasks["T2I"]["bases"][0]["categories"]
    category_names = [category["category"] for category in t2i_categories]
    assert "VAE" in category_names
    assert "VEA" not in category_names
    base_category = next(category for category in t2i_categories if category["category"] == "BaseModel")
    base_item = next(item for item in base_category["items"] if item["name"] == "realvisxl.safetensors")
    assert base_item["task_api"] == "text-to-image"
    assert base_item["apply_supported"] is True

    v2v_category = tasks["V2V"]["bases"][0]["categories"][0]
    v2v_item = v2v_category["items"][0]
    assert v2v_item["task_api"] == "image-to-video"


def test_local_models_tree_includes_legacy_root_layout(client: TestClient, tmp_path: Path) -> None:
    models_dir = tmp_path / "models_legacy_root"
    legacy_dir = models_dir / "legacy-download-model"
    legacy_dir.mkdir(parents=True)
    (legacy_dir / "model_index.json").write_text(json.dumps({"_class_name": "StableDiffusionXLPipeline"}), encoding="utf-8")
    (legacy_dir / "videogen_meta.json").write_text(
        json.dumps(
            {
                "source": "huggingface",
                "repo_id": "org/legacy-model",
                "task": "text-to-image",
                "base_model": "StableDiffusion XL",
                "category": "BaseModel",
            }
        ),
        encoding="utf-8",
    )

    resp = client.get("/api/models/local/tree", params={"dir": str(models_dir)})
    assert resp.status_code == 200
    tasks = {task["task"]: task for task in resp.json()["tasks"]}
    assert "T2I" in tasks
    imported_base = next((base for base in tasks["T2I"]["bases"] if base["base_name"] == "Imported"), None)
    assert imported_base is not None
    base_category = next((cat for cat in imported_base["categories"] if cat["category"] == "BaseModel"), None)
    assert base_category is not None
    assert any(item["name"] == "legacy-download-model" for item in base_category["items"])


def test_local_models_tree_rescan_reflects_new_item(client: TestClient, tmp_path: Path) -> None:
    models_dir = tmp_path / "models_tree_rescan"
    base_dir = models_dir / "T2I" / "SD1" / "BaseModel"
    base_dir.mkdir(parents=True)
    (base_dir / "initial.safetensors").write_bytes(b"old")

    first = client.get("/api/models/local/tree", params={"dir": str(models_dir)})
    assert first.status_code == 200
    first_names = {item["name"] for item in first.json()["flat_items"]}
    assert "initial.safetensors" in first_names
    assert "added.safetensors" not in first_names

    (base_dir / "added.safetensors").write_bytes(b"new")
    rescan = client.post("/api/models/local/rescan", json={"dir": str(models_dir)})
    assert rescan.status_code == 200
    names_after_rescan = {item["name"] for item in rescan.json()["flat_items"]}
    assert "added.safetensors" in names_after_rescan


def test_local_models_reveal_endpoint(client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    models_dir = tmp_path / "models_reveal"
    model_dir = models_dir / "T2I" / "SDXL1.0" / "BaseModel" / "demo-model"
    model_dir.mkdir(parents=True)
    (model_dir / "model_index.json").write_text(json.dumps({"_class_name": "StableDiffusionXLPipeline"}), encoding="utf-8")

    if os.name == "nt":
        calls: list[list[str]] = []

        def fake_popen(args, shell=False):  # type: ignore[override]
            calls.append(list(args))
            return object()

        monkeypatch.setattr(main.subprocess, "Popen", fake_popen)
        resp = client.post("/api/models/local/reveal", json={"path": str(model_dir), "base_dir": str(models_dir)})
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"
        assert calls
        assert calls[0][0].lower() == "explorer"
    else:
        resp = client.post("/api/models/local/reveal", json={"path": str(model_dir), "base_dir": str(models_dir)})
        assert resp.status_code == 200
        assert resp.json()["status"] == "not_supported"


def test_tasks_list_endpoint_filters_and_orders(client: TestClient) -> None:
    t1 = main.create_task("download", "queued")
    t2 = main.create_task("download", "queued")
    main.update_task(t1, status="running", progress=0.3, message="Downloading")
    time.sleep(0.01)
    main.update_task(t2, status="completed", progress=1.0, message="Download complete")

    resp_all = client.get("/api/tasks", params={"task_type": "download", "status": "all", "limit": 10})
    assert resp_all.status_code == 200
    tasks_all = resp_all.json()["tasks"]
    assert any(task["id"] == t1 for task in tasks_all)
    assert any(task["id"] == t2 for task in tasks_all)

    resp_running = client.get("/api/tasks", params={"task_type": "download", "status": "running", "limit": 10})
    assert resp_running.status_code == 200
    running_ids = {task["id"] for task in resp_running.json()["tasks"]}
    assert t1 in running_ids
    assert t2 not in running_ids


def test_text2image_task_lifecycle_and_image_endpoint(client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    outputs_dir = tmp_path / "outputs"
    models_dir = tmp_path / "models"
    tmp_dir = tmp_path / "tmp"
    settings = client.get("/api/settings").json()
    settings["paths"]["outputs_dir"] = str(outputs_dir.resolve())
    settings["paths"]["models_dir"] = str(models_dir.resolve())
    settings["paths"]["tmp_dir"] = str(tmp_dir.resolve())
    put_resp = client.put("/api/settings", json=settings)
    assert put_resp.status_code == 200

    class FakeText2ImagePipeline:
        def __call__(
            self,
            prompt: str,
            negative_prompt: str | None = None,
            num_inference_steps: int = 30,
            guidance_scale: float = 7.5,
            width: int = 512,
            height: int = 512,
            generator: object | None = None,
        ) -> object:
            image = Image.new("RGB", (width, height), color=(12, 34, 56))
            return type("FakeOut", (), {"images": [image]})()

    monkeypatch.setattr(main, "TORCH_IMPORT_ERROR", None)
    monkeypatch.setattr(main, "get_device_and_dtype", lambda: ("cpu", "float32"))
    monkeypatch.setattr(main, "get_pipeline", lambda kind, model_ref, settings_payload: FakeText2ImagePipeline())

    post = client.post(
        "/api/generate/text2image",
        json={
            "prompt": "test prompt",
            "negative_prompt": "",
            "model_id": "",
            "num_inference_steps": 4,
            "guidance_scale": 7.5,
            "width": 320,
            "height": 256,
        },
    )
    assert post.status_code == 200
    task_id = post.json()["task_id"]

    deadline = time.time() + 5
    last = None
    while time.time() < deadline:
        status = client.get(f"/api/tasks/{task_id}")
        assert status.status_code == 200
        last = status.json()
        if last["status"] in ("completed", "error"):
            break
        time.sleep(0.1)

    assert last is not None
    assert last["status"] == "completed"
    image_file = last["result"]["image_file"]
    image_path = outputs_dir / image_file
    assert image_path.exists()

    image_resp = client.get(f"/api/images/{image_file}")
    assert image_resp.status_code == 200
    assert image_resp.headers["content-type"].startswith("image/png")


def test_image2image_task_lifecycle(client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    outputs_dir = tmp_path / "outputs_i2i"
    models_dir = tmp_path / "models_i2i"
    tmp_dir = tmp_path / "tmp_i2i"
    settings = client.get("/api/settings").json()
    settings["paths"]["outputs_dir"] = str(outputs_dir.resolve())
    settings["paths"]["models_dir"] = str(models_dir.resolve())
    settings["paths"]["tmp_dir"] = str(tmp_dir.resolve())
    settings["defaults"]["image2image_model"] = "runwayml/stable-diffusion-v1-5"
    put_resp = client.put("/api/settings", json=settings)
    assert put_resp.status_code == 200

    class FakeImage2ImagePipeline:
        def __call__(
            self,
            prompt: str,
            image: Image.Image,
            negative_prompt: str | None = None,
            num_inference_steps: int = 30,
            guidance_scale: float = 7.5,
            strength: float = 0.6,
            generator: object | None = None,
        ) -> object:
            result = Image.new("RGB", image.size, color=(77, 99, 11))
            return type("FakeOut", (), {"images": [result]})()

    monkeypatch.setattr(main, "TORCH_IMPORT_ERROR", None)
    monkeypatch.setattr(main, "get_device_and_dtype", lambda: ("cpu", "float32"))
    monkeypatch.setattr(main, "get_pipeline", lambda kind, model_ref, settings_payload: FakeImage2ImagePipeline())

    input_image = tmp_path / "input.png"
    Image.new("RGB", (256, 128), color=(1, 2, 3)).save(input_image)
    with input_image.open("rb") as handle:
        post = client.post(
            "/api/generate/image2image",
            files={"image": ("input.png", handle, "image/png")},
            data={
                "prompt": "refine this image",
                "negative_prompt": "",
                "model_id": "",
                "num_inference_steps": "6",
                "guidance_scale": "7.5",
                "strength": "0.55",
                "width": "256",
                "height": "128",
            },
        )
    assert post.status_code == 200
    task_id = post.json()["task_id"]

    deadline = time.time() + 5
    last = None
    while time.time() < deadline:
        status = client.get(f"/api/tasks/{task_id}")
        assert status.status_code == 200
        last = status.json()
        if last["status"] in ("completed", "error"):
            break
        time.sleep(0.1)

    assert last is not None
    assert last["status"] == "completed"
    assert "image_file" in last["result"]


def test_lora_catalog_filters_by_base_model(client: TestClient, tmp_path: Path) -> None:
    models_dir = tmp_path / "models_lora"
    models_dir.mkdir(parents=True)
    lora_dir = models_dir / "author--style-lora"
    lora_dir.mkdir(parents=True)
    (lora_dir / "adapter_config.json").write_text(
        json.dumps({"base_model_name_or_path": "runwayml/stable-diffusion-v1-5"}),
        encoding="utf-8",
    )
    (lora_dir / "pytorch_lora_weights.safetensors").write_bytes(b"123")
    other_lora = models_dir / "author--other-lora"
    other_lora.mkdir(parents=True)
    (other_lora / "adapter_config.json").write_text(
        json.dumps({"base_model_name_or_path": "foo/bar"}),
        encoding="utf-8",
    )
    (other_lora / "pytorch_lora_weights.safetensors").write_bytes(b"456")

    settings = client.get("/api/settings").json()
    settings["paths"]["models_dir"] = str(models_dir.resolve())
    settings["defaults"]["text2image_model"] = "runwayml/stable-diffusion-v1-5"
    put_resp = client.put("/api/settings", json=settings)
    assert put_resp.status_code == 200

    resp = client.get("/api/models/loras/catalog", params={"task": "text-to-image", "model_ref": "runwayml/stable-diffusion-v1-5"})
    assert resp.status_code == 200
    ids = {item["id"] for item in resp.json()["items"]}
    assert "author/style-lora" in ids
    assert "author/other-lora" not in ids


def test_lora_catalog_allows_unknown_video_lora_for_manual_selection(client: TestClient, tmp_path: Path) -> None:
    models_dir = tmp_path / "models_video_lora_filter"
    models_dir.mkdir(parents=True)
    # No adapter_config.json, so this LoRA has no explicit base hint.
    lora_dir = models_dir / "author--lcm-lora-sdxl"
    lora_dir.mkdir(parents=True)
    (lora_dir / "pytorch_lora_weights.safetensors").write_bytes(b"123")

    settings = client.get("/api/settings").json()
    settings["paths"]["models_dir"] = str(models_dir.resolve())
    settings["defaults"]["text2video_model"] = "damo-vilab/text-to-video-ms-1.7b"
    put_resp = client.put("/api/settings", json=settings)
    assert put_resp.status_code == 200

    resp = client.get("/api/models/loras/catalog", params={"task": "text-to-video", "model_ref": "damo-vilab/text-to-video-ms-1.7b"})
    assert resp.status_code == 200
    ids = {item["id"] for item in resp.json()["items"]}
    assert "author/lcm-lora-sdxl" in ids


def test_text2video_worker_skips_incompatible_lora(client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    outputs_dir = tmp_path / "outputs_t2v"
    models_dir = tmp_path / "models_t2v"
    tmp_dir = tmp_path / "tmp_t2v"
    settings = client.get("/api/settings").json()
    settings["paths"]["outputs_dir"] = str(outputs_dir.resolve())
    settings["paths"]["models_dir"] = str(models_dir.resolve())
    settings["paths"]["tmp_dir"] = str(tmp_dir.resolve())
    settings["defaults"]["text2video_model"] = "damo-vilab/text-to-video-ms-1.7b"
    put_resp = client.put("/api/settings", json=settings)
    assert put_resp.status_code == 200

    class FakeText2VideoPipeline:
        def load_lora_weights(self, *_args, **_kwargs):
            raise AttributeError("'UNet3DConditionModel' object has no attribute 'load_lora_adapter'")

        def __call__(self, **_kwargs):
            frames = [np.zeros((16, 16, 3), dtype=np.uint8) for _ in range(8)]
            return type("FakeOut", (), {"frames": [frames]})()

    def fake_export(_frames, output_path: Path, fps: int = 0, **_kwargs) -> str:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"\x00\x00\x00\x18ftypmp42")
        return "fake_codec"

    monkeypatch.setattr(main, "get_pipeline", lambda kind, model_ref, settings_payload: FakeText2VideoPipeline())
    monkeypatch.setattr(main, "get_device_and_dtype", lambda: ("cuda", "float16"))
    monkeypatch.setattr(main, "export_video_with_fallback", fake_export)

    task_id = main.create_task("text2video", "Generation queued")
    payload = main.Text2VideoRequest(
        prompt="test prompt",
        model_id="",
        lora_ids=["author/lcm-lora-sdxl"],
        num_inference_steps=2,
        num_frames=8,
        guidance_scale=9.0,
        fps=8,
        seed=None,
        backend="cuda",
    )

    main.text2video_worker(task_id, payload)
    last = client.get(f"/api/tasks/{task_id}").json()
    assert last["status"] == "completed"
    assert last["result"]["loras"] == []
    assert (outputs_dir / last["result"]["video_file"]).exists()


def test_image2video_accepts_duration_seconds_form_value(client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    def fake_worker(task_id: str, payload: dict[str, object]) -> None:
        captured["task_id"] = task_id
        captured["payload"] = dict(payload)

    monkeypatch.setattr(main, "image2video_worker", fake_worker)

    image_path = tmp_path / "input.png"
    Image.new("RGB", (16, 16), color=(10, 20, 30)).save(image_path)
    with image_path.open("rb") as handle:
        resp = client.post(
            "/api/generate/image2video",
            data={
                "prompt": "test prompt",
                "negative_prompt": "",
                "model_id": "",
                "lora_id": "",
                "num_inference_steps": "2",
                "duration_seconds": "12.5",
                "guidance_scale": "9.0",
                "fps": "8",
                "width": "256",
                "height": "256",
            },
            files={"image": ("input.png", handle, "image/png")},
        )
    assert resp.status_code == 200

    deadline = time.time() + 2
    while time.time() < deadline and "payload" not in captured:
        time.sleep(0.05)

    assert "payload" in captured
    payload = captured["payload"]
    assert isinstance(payload, dict)
    assert float(payload["duration_seconds"]) == 12.5
    assert payload["num_frames"] is None


def test_recent_logs_endpoint(client: TestClient) -> None:
    resp = client.get("/api/logs/recent", params={"lines": 50})
    assert resp.status_code == 200
    body = resp.json()
    assert "log_file" in body
    assert Path(body["log_file"]).exists()
    assert isinstance(body.get("lines"), list)


def test_clear_hf_cache_endpoint(client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    hub = tmp_path / "hub"
    transformers = tmp_path / "transformers"
    hub.mkdir(parents=True)
    transformers.mkdir(parents=True)
    (hub / "x.bin").write_bytes(b"123")
    (transformers / "y.bin").write_bytes(b"456")

    monkeypatch.setattr(main, "gather_hf_cache_candidates", lambda: {hub, transformers})

    resp = client.post("/api/cache/hf/clear", json={"dry_run": False})
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert str(hub.resolve()) in body["removed_paths"]
    assert str(transformers.resolve()) in body["removed_paths"]
    assert not hub.exists()
    assert not transformers.exists()


def test_runtime_endpoint(client: TestClient) -> None:
    resp = client.get("/api/runtime")
    assert resp.status_code == 200
    body = resp.json()
    assert "rocm_aotriton_env" in body
    assert "preferred_dtype" in body
    assert "hardware_profile" in body
    assert "load_policy_preview" in body


def test_cancel_task_endpoint(client: TestClient) -> None:
    task_id = main.create_task("download", "queued")
    main.update_task(task_id, status="running", progress=0.1, message="running")
    resp = client.post("/api/tasks/cancel", json={"task_id": task_id})
    assert resp.status_code == 200
    task = client.get(f"/api/tasks/{task_id}")
    assert task.status_code == 200
    assert task.json()["cancel_requested"] is True


def test_cleanup_endpoint_removes_expired_files(client: TestClient, tmp_path: Path) -> None:
    outputs_dir = tmp_path / "cleanup_outputs"
    tmp_dir = tmp_path / "cleanup_tmp"
    outputs_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir.mkdir(parents=True, exist_ok=True)
    stale_output = outputs_dir / "stale.png"
    stale_tmp = tmp_dir / "stale.tmp"
    stale_output.write_bytes(b"123")
    stale_tmp.write_bytes(b"456")
    old_ts = time.time() - (10 * 24 * 60 * 60)
    os.utime(stale_output, (old_ts, old_ts))
    os.utime(stale_tmp, (old_ts, old_ts))

    settings = client.get("/api/settings").json()
    settings["paths"]["outputs_dir"] = str(outputs_dir.resolve())
    settings["paths"]["tmp_dir"] = str(tmp_dir.resolve())
    settings["storage"]["cleanup_max_age_days"] = 7
    settings["storage"]["cleanup_max_outputs_count"] = 100
    settings["storage"]["cleanup_max_tmp_count"] = 100
    put_resp = client.put("/api/settings", json=settings)
    assert put_resp.status_code == 200

    resp = client.post("/api/cleanup", json={"include_cache": False})
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert not stale_output.exists()
    assert not stale_tmp.exists()
