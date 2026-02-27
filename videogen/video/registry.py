from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Literal

VideoTask = Literal["text-to-video", "image-to-video"]
ModelKind = Literal["t2v", "i2v", "hybrid"]
SupportLevel = Literal["ready", "limited", "requires_patch", "not_supported"]


@dataclass(frozen=True)
class VideoModelSpec:
    """Unified model metadata shown to API/UI.

    Notes:
    - This object is intentionally static and lightweight.
    - Runtime-specific capability checks are layered on top in API responses.
    """

    key: str
    display_name: str
    kind: ModelKind
    support_level: SupportLevel
    tasks: tuple[VideoTask, ...]
    required_inputs: tuple[str, ...]
    recommended_dtype: str
    rocm_notes: str
    supported_resolution_max: tuple[int, int] = (1280, 720)
    supported_duration_sec_max: float = 16.0
    supported_num_frames_max: int = 128
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "key": self.key,
            "display_name": self.display_name,
            "kind": self.kind,
            "support_level": self.support_level,
            "tasks": list(self.tasks),
            "required_inputs": list(self.required_inputs),
            "recommended_dtype": self.recommended_dtype,
            "rocm_notes": self.rocm_notes,
            "supported_resolution_max": {"width": int(self.supported_resolution_max[0]), "height": int(self.supported_resolution_max[1])},
            "supported_duration_sec_max": float(self.supported_duration_sec_max),
            "supported_num_frames_max": int(self.supported_num_frames_max),
            "extra": dict(self.extra),
        }


# Phase 2: high-confidence adapters
# Phase 3: expose as limited/requires_patch until adapter path is validated on ROCm.
VIDEO_MODEL_REGISTRY: Dict[str, VideoModelSpec] = {
    "text2videosd": VideoModelSpec(
        key="text2videosd",
        display_name="TextToVideoSDPipeline",
        kind="t2v",
        support_level="ready",
        tasks=("text-to-video",),
        required_inputs=("prompt",),
        recommended_dtype="bf16",
        rocm_notes="ROCmで比較的安定。長尺はframepack推奨。",
        supported_resolution_max=(1024, 576),
        supported_duration_sec_max=20.0,
        supported_num_frames_max=192,
    ),
    "wan": VideoModelSpec(
        key="wan",
        display_name="WanPipeline / WanImageToVideoPipeline",
        kind="hybrid",
        support_level="ready",
        tasks=("text-to-video", "image-to-video"),
        required_inputs=("prompt",),
        recommended_dtype="bf16",
        rocm_notes="VRAMが十分な場合はfull VRAM loadを推奨。非Diffusers repoは自動で -Diffusers を探索。",
        supported_resolution_max=(1280, 720),
        supported_duration_sec_max=30.0,
        supported_num_frames_max=240,
    ),
    "cogvideox": VideoModelSpec(
        key="cogvideox",
        display_name="CogVideoX",
        kind="hybrid",
        support_level="ready",
        tasks=("text-to-video", "image-to-video"),
        required_inputs=("prompt",),
        recommended_dtype="bf16",
        rocm_notes="モデルサイズが大きいため96GBクラスではfull VRAM、低VRAMではauto mapへフォールバック。",
        supported_resolution_max=(1280, 720),
        supported_duration_sec_max=16.0,
        supported_num_frames_max=128,
    ),
    "stablevideodiffusion": VideoModelSpec(
        key="stablevideodiffusion",
        display_name="StableVideoDiffusionPipeline",
        kind="i2v",
        support_level="ready",
        tasks=("image-to-video",),
        required_inputs=("image",),
        recommended_dtype="float16",
        rocm_notes="Image-to-Video専用。promptではなく入力画像を条件に生成。",
        supported_resolution_max=(1024, 576),
        supported_duration_sec_max=8.0,
        supported_num_frames_max=64,
    ),
    "ltxvideo": VideoModelSpec(
        key="ltxvideo",
        display_name="LTX-Video",
        kind="hybrid",
        support_level="limited",
        tasks=("text-to-video", "image-to-video"),
        required_inputs=("prompt",),
        recommended_dtype="bf16",
        rocm_notes="Phase 3: 一部引数差分を吸収済みだが、モデル依存差分は要検証。",
    ),
    "hunyuanvideo": VideoModelSpec(
        key="hunyuanvideo",
        display_name="HunyuanVideo",
        kind="hybrid",
        support_level="limited",
        tasks=("text-to-video", "image-to-video"),
        required_inputs=("prompt",),
        recommended_dtype="bf16",
        rocm_notes="Phase 3: API差分が大きいため adapter fallback で運用。",
    ),
    "sanavideo": VideoModelSpec(
        key="sanavideo",
        display_name="SanaVideo",
        kind="hybrid",
        support_level="requires_patch",
        tasks=("text-to-video", "image-to-video"),
        required_inputs=("prompt",),
        recommended_dtype="bf16",
        rocm_notes="Phase 3: upstream pipeline実装差分が大きく要パッチ。",
    ),
    "animatediff": VideoModelSpec(
        key="animatediff",
        display_name="AnimateDiff",
        kind="hybrid",
        support_level="limited",
        tasks=("text-to-video", "image-to-video"),
        required_inputs=("prompt",),
        recommended_dtype="float16",
        rocm_notes="ROCmでは速度より安定性優先。高解像度は非推奨。",
    ),
}


def model_specs_for_task(task: VideoTask) -> list[VideoModelSpec]:
    return [spec for spec in VIDEO_MODEL_REGISTRY.values() if task in spec.tasks]
