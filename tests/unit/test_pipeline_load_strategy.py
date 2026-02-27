from types import SimpleNamespace

import pytest

import main

pytestmark = pytest.mark.unit


def test_full_vram_fallback_omits_device_map_and_resets_before_to(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []

    class FakePipe:
        def __init__(self) -> None:
            self.hf_device_map = {"": "cuda:0"}
            self.reset_calls = 0
            self.to_calls = 0

        def reset_device_map(self) -> None:
            self.reset_calls += 1
            self.hf_device_map = {}

        def to(self, _device: object) -> "FakePipe":
            self.to_calls += 1
            if self.hf_device_map:
                raise ValueError("device mapping strategy")
            return self

    def fake_loader(_source: str, **kwargs: object) -> FakePipe:
        calls.append(dict(kwargs))
        if isinstance(kwargs.get("device_map"), dict):
            raise ValueError("`device_map` must be a string.")
        return FakePipe()

    fake_hw = SimpleNamespace(
        cuda_available=True,
        gpu_total_bytes=96 * 1024**3,
        to_dict=lambda: {"gpu_total_bytes": 96 * 1024**3},
    )
    fake_torch = SimpleNamespace(
        device=lambda name: name,
        cuda=SimpleNamespace(is_available=lambda: True),
    )

    monkeypatch.setattr(main, "torch", fake_torch)
    monkeypatch.setattr(main, "TORCH_IMPORT_ERROR", None)
    monkeypatch.setattr(main, "detect_hardware_profile", lambda **_kwargs: fake_hw)
    monkeypatch.setattr(main, "resolve_safetensors_mode", lambda _settings: (True, "prefer_safetensors"))
    monkeypatch.setattr(main, "verify_safetensors_integrity", lambda _source, _settings: {"failed": 0, "checked": 0, "failures": []})
    monkeypatch.setattr(main, "is_96gb_vram_profile", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(main, "build_loader_max_memory", lambda _settings, _hardware: {0: "90GB", "cpu": "4GB"})
    monkeypatch.setattr(main, "pipeline_meta_tensor_names", lambda _pipe: [])
    monkeypatch.setattr(main, "should_disable_vae_tiling", lambda _settings: False)
    monkeypatch.setattr(main, "log_memory_snapshot", lambda *_args, **_kwargs: None)

    pipe = main.load_pipeline_from_pretrained_with_strategy(
        loader=fake_loader,
        source="dummy/model",
        dtype="float16",
        prefer_gpu_device_map=True,
        kind="text-to-video",
        settings={"server": {}},
        cache_key="text-to-video:dummy/model",
        status_callback=None,
    )

    assert len(calls) == 2
    assert isinstance(calls[0].get("device_map"), dict)
    assert "device_map" not in calls[1]
    assert getattr(pipe, "reset_calls", 0) >= 1
    assert getattr(pipe, "to_calls", 0) >= 1
