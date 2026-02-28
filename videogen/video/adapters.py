from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Literal, Optional

from .registry import VIDEO_MODEL_REGISTRY, VideoModelSpec

VideoTask = Literal["text-to-video", "image-to-video"]


@dataclass
class VideoPipelineAdapter:
    """Common adapter interface for heterogeneous diffusers video pipelines."""

    key: str
    spec: VideoModelSpec
    last_extract_route: str = "unknown"
    last_extract_summary: str = ""

    def load(
        self,
        *,
        loader: Any,
        source: str,
        load_with_strategy: Any,
        dtype: Any,
        kind: VideoTask,
        settings: Dict[str, Any],
        cache_key: str,
        status_callback: Optional[Any],
    ) -> Any:
        return load_with_strategy(
            loader=lambda src, **kwargs: loader.from_pretrained(src, **kwargs),
            source=source,
            dtype=dtype,
            prefer_gpu_device_map=True,
            kind=kind,
            settings=settings,
            cache_key=cache_key,
            status_callback=status_callback,
        )

    def prepare_inputs(
        self,
        *,
        task: VideoTask,
        payload: Dict[str, Any],
        num_frames: int,
        generator: Optional[Any],
        step_progress_kwargs: Dict[str, Any],
        lora_scale: float,
        current_image: Optional[Any] = None,
        framepack_context_arg: str = "",
        carry_image: Optional[Any] = None,
    ) -> Dict[str, Any]:
        if task == "image-to-video":
            return self._prepare_i2v_inputs(
                payload=payload,
                num_frames=num_frames,
                generator=generator,
                step_progress_kwargs=step_progress_kwargs,
                lora_scale=lora_scale,
                current_image=current_image,
            )
        return self._prepare_t2v_inputs(
            payload=payload,
            num_frames=num_frames,
            generator=generator,
            step_progress_kwargs=step_progress_kwargs,
            lora_scale=lora_scale,
            framepack_context_arg=framepack_context_arg,
            carry_image=carry_image,
        )

    def _prepare_t2v_inputs(
        self,
        *,
        payload: Dict[str, Any],
        num_frames: int,
        generator: Optional[Any],
        step_progress_kwargs: Dict[str, Any],
        lora_scale: float,
        framepack_context_arg: str,
        carry_image: Optional[Any],
    ) -> Dict[str, Any]:
        kwargs: Dict[str, Any] = {
            "prompt": payload["prompt"],
            "negative_prompt": payload.get("negative_prompt") or None,
            "num_inference_steps": int(payload["num_inference_steps"]),
            "num_frames": int(num_frames),
            "guidance_scale": float(payload["guidance_scale"]),
            "generator": generator,
            "cross_attention_kwargs": {"scale": lora_scale} if int(payload.get("_lora_count") or 0) == 1 else None,
            **step_progress_kwargs,
        }
        if framepack_context_arg and carry_image is not None:
            kwargs[framepack_context_arg] = carry_image
        return kwargs

    def _prepare_i2v_inputs(
        self,
        *,
        payload: Dict[str, Any],
        num_frames: int,
        generator: Optional[Any],
        step_progress_kwargs: Dict[str, Any],
        lora_scale: float,
        current_image: Optional[Any],
    ) -> Dict[str, Any]:
        return {
            "prompt": payload["prompt"],
            "negative_prompt": payload.get("negative_prompt") or None,
            "image": current_image,
            "height": int(payload["height"]),
            "width": int(payload["width"]),
            "target_fps": int(payload["fps"]),
            "fps": int(payload["fps"]),
            "num_inference_steps": int(payload["num_inference_steps"]),
            "num_frames": int(num_frames),
            "guidance_scale": float(payload["guidance_scale"]),
            "generator": generator,
            "cross_attention_kwargs": {"scale": lora_scale} if int(payload.get("_lora_count") or 0) == 1 else None,
            **step_progress_kwargs,
        }

    def _record_extract_route(self, route: str, output: Any) -> None:
        self.last_extract_route = str(route or "unknown")
        self.last_extract_summary = self._output_debug_summary(output)

    def _is_frame_like(self, value: Any) -> bool:
        if value is None:
            return False
        if isinstance(value, (str, bytes, bytearray, dict)):
            return False
        module_name = str(getattr(type(value), "__module__", "")).lower()
        class_name = str(getattr(type(value), "__name__", "")).lower()
        if "pil" in module_name and "image" in class_name:
            return True
        ndim = getattr(value, "ndim", None)
        if isinstance(ndim, int):
            return 2 <= int(ndim) <= 4
        shape = getattr(value, "shape", None)
        if isinstance(shape, tuple):
            return 2 <= len(shape) <= 4
        return False

    def _extract_frame_list_candidate(self, value: Any, route: str, depth: int = 0) -> Optional[list[Any]]:
        if value is None:
            return None
        if depth > 4:
            return None

        # Frame sequence: [frame0, frame1, ...]
        if isinstance(value, (list, tuple)):
            sequence = list(value)
            if not sequence:
                return []
            first = sequence[0]
            if self._is_frame_like(first):
                return sequence
            # Nested list from some pipelines: [[frame0, frame1, ...]]
            nested = self._extract_frame_list_candidate(first, f"{route}[0]", depth + 1)
            if nested is not None and len(nested) > 0:
                return nested
            # list/tuple can also be a batched tensor-like structure represented by arrays
            extracted = self._extract_frames_from_videos(value)
            if extracted:
                return extracted
            return None

        # Tensor / ndarray-like container (e.g. output.video / output.videos)
        extracted = self._extract_frames_from_videos(value)
        if extracted:
            return extracted

        if self._is_frame_like(value):
            return [value]
        return None

    def _output_debug_summary(self, output: Any) -> str:
        output_type = f"{type(output).__module__}.{type(output).__name__}"
        attrs = [name for name in dir(output) if not name.startswith("_")]
        attrs_preview = ", ".join(attrs[:40])
        extra: list[str] = []
        if isinstance(output, (list, tuple)):
            extra.append(f"len={len(output)}")
            if output:
                extra.append(f"first_type={type(output[0]).__module__}.{type(output[0]).__name__}")
        return f"type={output_type}; attrs=[{attrs_preview}]" + (f"; {', '.join(extra)}" if extra else "")

    def extract_frames(self, output: Any) -> list[Any]:
        """
        Extract frames from heterogeneous pipeline outputs.

        Why this shape:
        - Wan/Cog/SVD/Diffusers versions return different output containers.
        - We resolve by capability order (frames/video/videos/list/getitem) instead of
          model-specific branching so future pipeline variants keep working.
        """
        if output is None:
            self._record_extract_route("none", output)
            raise RuntimeError("Pipeline output is None; cannot extract frames.")

        # 1) out.frames
        frames_attr = getattr(output, "frames", None)
        if frames_attr is not None:
            frames = self._extract_frame_list_candidate(frames_attr, "frames")
            if frames:
                self._record_extract_route("frames", output)
                return frames

        # 2) out.video
        video_attr = getattr(output, "video", None)
        if video_attr is not None:
            frames = self._extract_frame_list_candidate(video_attr, "video")
            if frames:
                self._record_extract_route("video", output)
                return frames

        # 3) out.videos
        videos_attr = getattr(output, "videos", None)
        if videos_attr is not None:
            frames = self._extract_frame_list_candidate(videos_attr, "videos")
            if frames:
                self._record_extract_route("videos", output)
                return frames

        # 4) output itself list/tuple
        if isinstance(output, (list, tuple)):
            frames = self._extract_frame_list_candidate(output, "list")
            if frames:
                self._record_extract_route("list", output)
                return frames
            if len(output) > 0:
                nested = self._extract_frame_list_candidate(output[0], "list[0]")
                if nested:
                    self._record_extract_route("list[0]", output)
                    return nested

        # 5) generic indexable fallback: out[0]
        if hasattr(output, "__getitem__"):
            with_index = None
            try:
                with_index = output[0]
            except Exception:
                with_index = None
            if with_index is not None:
                frames = self._extract_frame_list_candidate(with_index, "getitem[0]")
                if frames:
                    self._record_extract_route("getitem[0]", output)
                    return frames

        self._record_extract_route("unresolved", output)
        raise RuntimeError("Failed to extract frames from pipeline output. " f"{self._output_debug_summary(output)}")

    @staticmethod
    def _to_numpy(value: Any) -> Any:
        converted = value
        if hasattr(converted, "detach"):
            converted = converted.detach()
        if hasattr(converted, "cpu"):
            converted = converted.cpu()
        if hasattr(converted, "numpy"):
            return converted.numpy()
        try:
            import numpy as np

            return np.asarray(converted)
        except Exception:
            return converted

    def _extract_frames_from_videos(self, videos: Any) -> list[Any]:
        if videos is None:
            return []
        if isinstance(videos, list) and videos:
            first = videos[0]
            if isinstance(first, list):
                return list(first)
            if self._is_frame_like(first):
                return list(videos)
            nested = self._extract_frames_from_videos(first)
            if nested:
                return nested
            return list(videos)
        arr = self._to_numpy(videos)
        shape = tuple(getattr(arr, "shape", ()) or ())
        ndim = int(getattr(arr, "ndim", 0) or 0)
        if ndim == 3:
            if len(shape) == 3 and shape[-1] in (1, 3, 4):
                return [arr]
            if len(shape) == 3 and shape[0] in (1, 3, 4):
                return [arr.transpose(1, 2, 0)]
        if ndim == 5:
            # [B, F, H, W, C]
            if len(shape) == 5 and shape[-1] in (1, 3, 4):
                return [arr[0, i] for i in range(shape[1])]
            # [B, C, F, H, W]
            if len(shape) == 5 and shape[1] in (1, 3, 4):
                return [arr[0, :, i, :, :].transpose(1, 2, 0) for i in range(shape[2])]
            # [B, F, C, H, W]
            if len(shape) == 5 and shape[2] in (1, 3, 4):
                return [arr[0, i, :, :, :].transpose(1, 2, 0) for i in range(shape[1])]
            return []
        if ndim == 4:
            # [F, H, W, C]
            if len(shape) == 4 and shape[-1] in (1, 3, 4):
                return [arr[i] for i in range(shape[0])]
            # [F, C, H, W]
            if len(shape) == 4 and shape[1] in (1, 3, 4):
                return [arr[i].transpose(1, 2, 0) for i in range(shape[0])]
        return []


class CogVideoXAdapter(VideoPipelineAdapter):
    def _prepare_i2v_inputs(
        self,
        *,
        payload: Dict[str, Any],
        num_frames: int,
        generator: Optional[Any],
        step_progress_kwargs: Dict[str, Any],
        lora_scale: float,
        current_image: Optional[Any],
    ) -> Dict[str, Any]:
        kwargs = super()._prepare_i2v_inputs(
            payload=payload,
            num_frames=num_frames,
            generator=generator,
            step_progress_kwargs=step_progress_kwargs,
            lora_scale=lora_scale,
            current_image=current_image,
        )
        # CogVideoX variants sometimes expose fps instead of target_fps.
        kwargs.setdefault("fps", int(payload["fps"]))
        return kwargs


class StableVideoDiffusionAdapter(VideoPipelineAdapter):
    def _prepare_i2v_inputs(
        self,
        *,
        payload: Dict[str, Any],
        num_frames: int,
        generator: Optional[Any],
        step_progress_kwargs: Dict[str, Any],
        lora_scale: float,
        current_image: Optional[Any],
    ) -> Dict[str, Any]:
        # StableVideoDiffusionPipeline is image-conditioned and typically does
        # not consume prompt/negative_prompt. Keep only compatible args.
        guidance = float(payload["guidance_scale"])
        return {
            "image": current_image,
            "height": int(payload["height"]),
            "width": int(payload["width"]),
            "fps": int(payload["fps"]),
            "num_inference_steps": int(payload["num_inference_steps"]),
            "num_frames": int(num_frames),
            "min_guidance_scale": guidance,
            "max_guidance_scale": guidance,
            "generator": generator,
            **step_progress_kwargs,
        }


class UnsupportedVideoAdapter(VideoPipelineAdapter):
    def prepare_inputs(self, **kwargs: Any) -> Dict[str, Any]:
        raise RuntimeError(
            f"Model adapter '{self.key}' is not ready on this build. " "Please use TextToVideoSD, Wan, or CogVideoX families."
        )


def adapter_key_from_pipeline(pipe: Any, source: str, model_ref: str = "") -> str:
    class_name = str(getattr(pipe, "__class__", type("x", (), {})).__name__ or "").lower()
    ref_text = f"{source} {model_ref}".lower()
    if "wan" in class_name or "wan" in ref_text:
        return "wan"
    if "cogvideox" in class_name or "cogvideo" in ref_text:
        return "cogvideox"
    if "texttovideosd" in class_name:
        return "text2videosd"
    if "stablevideodiffusion" in class_name or "stable-video-diffusion" in ref_text or "img2vid" in ref_text:
        return "stablevideodiffusion"
    if "ltx" in class_name or "ltx" in ref_text:
        return "ltxvideo"
    if "hunyuan" in class_name or "hunyuan" in ref_text:
        return "hunyuanvideo"
    if "sana" in class_name or "sana" in ref_text:
        return "sanavideo"
    if "animatediff" in class_name or "animatediff" in ref_text:
        return "animatediff"
    return "text2videosd"


def resolve_adapter_for_pipeline(pipe: Any, source: str, model_ref: str = "") -> VideoPipelineAdapter:
    key = adapter_key_from_pipeline(pipe=pipe, source=source, model_ref=model_ref)
    spec = VIDEO_MODEL_REGISTRY.get(key, VIDEO_MODEL_REGISTRY["text2videosd"])
    if key == "cogvideox":
        return CogVideoXAdapter(key=key, spec=spec)
    if key == "stablevideodiffusion":
        return StableVideoDiffusionAdapter(key=key, spec=spec)
    if spec.support_level in {"ready", "limited"}:
        return VideoPipelineAdapter(key=key, spec=spec)
    return UnsupportedVideoAdapter(key=key, spec=spec)
