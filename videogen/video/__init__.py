"""Video generation foundation package.

This package provides a unified abstraction layer for T2V/I2V models.
"""

from .adapters import (
    VideoPipelineAdapter,
    resolve_adapter_for_pipeline,
)
from .registry import (
    VIDEO_MODEL_REGISTRY,
    VideoModelSpec,
    model_specs_for_task,
)

__all__ = [
    "VIDEO_MODEL_REGISTRY",
    "VideoModelSpec",
    "VideoPipelineAdapter",
    "model_specs_for_task",
    "resolve_adapter_for_pipeline",
]
