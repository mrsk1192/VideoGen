import json
from pathlib import Path

import pytest

import main

pytestmark = pytest.mark.unit


def test_sanitize_desanitize_roundtrip() -> None:
    repo_id = "org-name/model-name"
    sanitized = main.sanitize_repo_id(repo_id)
    restored = main.desanitize_repo_id(sanitized)
    assert sanitized == "org-name--model-name"
    assert restored == repo_id


def test_deep_merge_nested() -> None:
    base = {"a": {"x": 1, "y": 2}, "b": 3}
    updates = {"a": {"y": 9}, "c": 4}
    merged = main.deep_merge(base, updates)
    assert merged == {"a": {"x": 1, "y": 9}, "b": 3, "c": 4}
    assert base == {"a": {"x": 1, "y": 2}, "b": 3}


def test_parse_civitai_model_id() -> None:
    assert main.parse_civitai_model_id("civitai/12345") == 12345
    assert main.parse_civitai_model_id("CivitAI/987") == 987
    assert main.parse_civitai_model_id("foo/bar") is None
    assert main.parse_civitai_model_id("civitai/not-number") is None


def test_find_local_preview_relpath(tmp_path: Path) -> None:
    models_dir = tmp_path / "models"
    model_dir = models_dir / "foo--bar"
    model_dir.mkdir(parents=True)
    preview = model_dir / "thumbnail.png"
    preview.write_bytes(b"fake")
    rel = main.find_local_preview_relpath(model_dir=model_dir, models_dir=models_dir)
    assert rel == "foo--bar/thumbnail.png"


def test_is_local_lora_dir_ignores_base_pipeline_with_lora_named_file(tmp_path: Path) -> None:
    model_dir = tmp_path / "base-model"
    model_dir.mkdir(parents=True)
    (model_dir / "model_index.json").write_text(
        json.dumps({"_class_name": "StableDiffusionXLPipeline"}),
        encoding="utf-8",
    )
    (model_dir / "sd_xl_offset_example-lora_1.0.safetensors").write_bytes(b"abc")
    assert main.is_local_lora_dir(model_dir) is False


def test_cache_get_or_set_uses_cached_value(monkeypatch: pytest.MonkeyPatch) -> None:
    main.SEARCH_API_CACHE.clear()
    counter = {"calls": 0}

    def loader() -> dict:
        counter["calls"] += 1
        return {"value": counter["calls"]}

    first = main.cache_get_or_set("test", {"a": 1}, loader)
    second = main.cache_get_or_set("test", {"a": 1}, loader)
    assert first["value"] == 1
    assert second["value"] == 1
    assert counter["calls"] == 1


def test_normalize_local_tree_category_alias() -> None:
    assert main.normalize_local_tree_category("VAE") == "VAE"
    assert main.normalize_local_tree_category("VEA") == "VAE"
    assert main.normalize_local_tree_category("lora") == "Lora"
    assert main.normalize_local_tree_category("base-model") == "BaseModel"


def test_is_local_tree_item_directory_accepts_single_file_checkpoint(tmp_path: Path) -> None:
    base_model_dir = tmp_path / "single-file-base"
    base_model_dir.mkdir(parents=True)
    (base_model_dir / "model.safetensors").write_bytes(b"abc")
    assert main.is_local_tree_item_directory(base_model_dir, "BaseModel") is True


def test_download_model_kind_normalization() -> None:
    assert main.normalize_download_model_kind("lora") == "Lora"
    assert main.normalize_download_model_kind("VAE") == "VAE"
    assert main.normalize_download_model_kind("checkpoint") == "BaseModel"


def test_resolve_framepack_plan_enables_long_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("VIDEOGEN_FRAMEPACK_SEGMENT_FRAMES", raising=False)
    monkeypatch.delenv("VIDEOGEN_VIDEO_CHUNK_FRAMES", raising=False)
    monkeypatch.delenv("VIDEOGEN_FRAMEPACK_OVERLAP_FRAMES", raising=False)
    monkeypatch.delenv("VIDEOGEN_FRAMEPACK_LONG_SEGMENT_FRAMES", raising=False)

    # 30 minutes at 8fps => long-video mode threshold.
    plan = main.resolve_framepack_plan(total_frames=30 * 60 * 8, fps=8)
    assert plan["long_video_mode"] is True
    assert int(plan["segment_frames"]) <= int(main.FRAMEPACK_LONG_SEGMENT_FRAMES_DEFAULT)
    assert int(plan["overlap_frames"]) < int(plan["segment_frames"])
    assert int(plan["pack_count"]) >= 1


def test_iter_framepack_segments_preserves_total_frame_count() -> None:
    segments = main.iter_framepack_segments(total_frames=25, segment_frames=8, overlap_frames=2)
    assert len(segments) >= 1
    assert segments[0]["trim_head_frames"] == 0
    assert all(segment["request_frames"] <= 8 for segment in segments)
    assert sum(segment["append_frames"] for segment in segments) == 25


def test_call_with_supported_kwargs_recovers_from_wan_ftfy_name_error(monkeypatch: pytest.MonkeyPatch) -> None:
    class DummyPipe:
        def __init__(self) -> None:
            self.calls = 0

        def __call__(self, prompt: str):
            self.calls += 1
            if self.calls == 1:
                raise NameError("name 'ftfy' is not defined")
            return {"ok": True, "prompt": prompt}

    pipe = DummyPipe()
    monkeypatch.setattr(main, "try_patch_wan_ftfy_dependency", lambda _pipe: True)
    result = main.call_with_supported_kwargs(pipe, {"prompt": "hello"})
    assert result["ok"] is True
    assert pipe.calls == 2


def test_call_with_supported_kwargs_reports_missing_wan_ftfy_dependency(monkeypatch: pytest.MonkeyPatch) -> None:
    class DummyPipe:
        def __call__(self, prompt: str):
            raise NameError("name 'ftfy' is not defined")

    monkeypatch.setattr(main, "try_patch_wan_ftfy_dependency", lambda _pipe: False)
    with pytest.raises(RuntimeError, match="ftfy"):
        main.call_with_supported_kwargs(DummyPipe(), {"prompt": "hello"})


def test_is_gpu_oom_error_detects_wrapped_exception_chain() -> None:
    inner = RuntimeError("HIP out of memory. Tried to allocate 50.00 MiB")
    outer = RuntimeError("GPU-first pipeline loading failed and CPU-heavy fallback is disabled.")
    outer.__cause__ = inner
    assert main.is_gpu_oom_error(outer) is True
