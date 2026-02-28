import pytest

from videogen.video.adapters import resolve_adapter_for_pipeline
from videogen.video.foundation import choose_video_load_policy
from videogen.video.registry import model_specs_for_task

pytestmark = pytest.mark.unit


def test_model_specs_for_task_contains_core_models() -> None:
    t2v_keys = {spec.key for spec in model_specs_for_task("text-to-video")}
    i2v_keys = {spec.key for spec in model_specs_for_task("image-to-video")}
    assert "text2videosd" in t2v_keys
    assert "wan" in t2v_keys
    assert "cogvideox" in t2v_keys
    assert "wan" in i2v_keys
    assert "cogvideox" in i2v_keys
    assert "stablevideodiffusion" in i2v_keys


def test_choose_video_load_policy_prefers_full_vram_profile() -> None:
    runtime = {
        "vram_profile": "96gb_class",
        "selected_dtype": "bf16",
    }
    policy = choose_video_load_policy(runtime=runtime)
    assert policy.mode == "full_vram"
    assert policy.device_map == {"": "cuda:0"}
    assert policy.enable_cpu_offload is False


def test_resolve_adapter_for_pipeline_detects_cogvideox() -> None:
    class CogVideoXPipeline:
        pass

    fake_pipe = CogVideoXPipeline()
    adapter = resolve_adapter_for_pipeline(fake_pipe, source="THUDM/CogVideoX-5b", model_ref="THUDM/CogVideoX-5b")
    assert adapter.key == "cogvideox"
    assert adapter.spec.display_name.lower().startswith("cogvideox")


def test_adapter_extract_frames_accepts_videos_nested_list() -> None:
    class WanPipeline:
        pass

    class Out:
        videos = [[1, 2, 3]]

    adapter = resolve_adapter_for_pipeline(WanPipeline(), source="Wan-AI/Wan2.1-T2V-1.3B-Diffusers", model_ref="wan")
    frames = adapter.extract_frames(Out())
    assert frames == [1, 2, 3]


def test_adapter_extract_frames_accepts_video_attr() -> None:
    class WanPipeline:
        pass

    class Out:
        video = [[10, 20, 30]]

    adapter = resolve_adapter_for_pipeline(WanPipeline(), source="Wan-AI/Wan2.1-T2V-1.3B-Diffusers", model_ref="wan")
    frames = adapter.extract_frames(Out())
    assert frames == [10, 20, 30]
    assert adapter.last_extract_route == "video"


def test_adapter_extract_frames_fallback_getitem() -> None:
    class WanPipeline:
        pass

    class Out:
        def __getitem__(self, index: int):
            if index == 0:
                return [[7, 8]]
            raise IndexError(index)

    adapter = resolve_adapter_for_pipeline(WanPipeline(), source="Wan-AI/Wan2.1-T2V-1.3B-Diffusers", model_ref="wan")
    frames = adapter.extract_frames(Out())
    assert frames == [7, 8]
    assert adapter.last_extract_route == "getitem[0]"


def test_adapter_extract_frames_error_has_debug_fields() -> None:
    class WanPipeline:
        pass

    class Out:
        foo = "bar"

    adapter = resolve_adapter_for_pipeline(WanPipeline(), source="Wan-AI/Wan2.1-T2V-1.3B-Diffusers", model_ref="wan")
    with pytest.raises(RuntimeError) as exc:
        adapter.extract_frames(Out())
    message = str(exc.value)
    assert "Failed to extract frames" in message
    assert "type=" in message
    assert "attrs=" in message
