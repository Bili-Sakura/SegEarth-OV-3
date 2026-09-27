"""SegEarth-OV3 inference on native Hugging Face transformers SAM 3."""

from .loading import DEFAULT_MODEL_ID, load_sam3
from .pipeline import SegEarthOV3Pipeline, pipeline, register_pipeline
from .prompts import PromptSpec, normalize_text_prompts

register_pipeline()

try:
    from .segmentor import SegEarthOV3Segmentation
    from .change_detector import SegEarthOV3CDSeg
except Exception:  # mmseg is optional for the Hugging Face pipeline
    SegEarthOV3Segmentation = None
    SegEarthOV3CDSeg = None

__all__ = [
    "DEFAULT_MODEL_ID",
    "PromptSpec",
    "SegEarthOV3CDSeg",
    "SegEarthOV3Pipeline",
    "SegEarthOV3Segmentation",
    "load_sam3",
    "normalize_text_prompts",
    "pipeline",
    "register_pipeline",
]
