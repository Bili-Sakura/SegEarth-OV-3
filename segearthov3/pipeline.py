"""GitHub entry point for the Hub-standard SegEarth-OV3 pipeline.

The implementation lives in ``hub/pipeline.py`` so the same file can be
published to a Hugging Face model repo (``pipeline.py`` + ``config.json``).
This module only adds local loading helpers (``from_pretrained`` / ``pipeline``).
"""

from __future__ import annotations

from . import _hub as _hub_mod
from .loading import DEFAULT_MODEL_ID, load_sam3

_hub = _hub_mod.load_hub_pipeline_module()

SegEarthOV3Pipeline = _hub.SegEarthOV3Pipeline
load_any_image = _hub.load_any_image
register_pipeline = _hub.register_pipeline


@classmethod
def _from_pretrained(cls, model_id: str = DEFAULT_MODEL_ID, **kwargs):
    """Load ``facebook/sam3`` (or a local snapshot) and wrap this pipeline."""
    device = kwargs.pop("device", None)
    model_kwargs = kwargs.pop("model_kwargs", {})
    model, processor = load_sam3(model_id, device=device, **model_kwargs)
    return cls(
        model=model,
        tokenizer=getattr(processor, "tokenizer", None),
        image_processor=getattr(processor, "image_processor", None),
        processor=processor,
        device=device,
        **kwargs,
    )


if getattr(SegEarthOV3Pipeline, "from_pretrained", None) is None:
    SegEarthOV3Pipeline.from_pretrained = _from_pretrained


def pipeline(
    model: str = DEFAULT_MODEL_ID,
    text_prompts: list | None = None,
    **kwargs,
) -> SegEarthOV3Pipeline:
    """Build the custom SegEarth-OV3 inference pipeline.

    Parameters
    ----------
    model:
        Hugging Face model id or local folder. Defaults to ``facebook/sam3``.
    text_prompts:
        List of class names or synonym groups, e.g.
        ``["background", ["building", "house"], "road"]``.
    """
    register_pipeline()
    return SegEarthOV3Pipeline.from_pretrained(
        model, text_prompts=text_prompts, **kwargs
    )
