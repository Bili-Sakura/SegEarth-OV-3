"""Custom Hugging Face pipeline for SegEarth-OV3 open-vocabulary segmentation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .engine import Sam3OVSSEngine
from .loading import DEFAULT_MODEL_ID, load_sam3
from .prompts import PromptSpec, normalize_text_prompts

try:
    from transformers import Pipeline
    from transformers.pipelines import PIPELINE_REGISTRY
except ImportError:  # allow engine-only use and unit tests without transformers
    Pipeline = object  # type: ignore[misc,assignment]
    PIPELINE_REGISTRY = None


def load_any_image(image: Any) -> Image.Image:
    """Load a PIL RGB image from a path, URL, array, or existing PIL image."""
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    if isinstance(image, (str, Path)):
        try:
            from transformers.image_utils import load_image

            return load_image(str(image)).convert("RGB")
        except Exception:
            return Image.open(image).convert("RGB")
    if isinstance(image, np.ndarray):
        if image.ndim == 3 and image.shape[0] in (1, 3) and image.shape[-1] not in (1, 3):
            image = np.transpose(image, (1, 2, 0))
        if image.ndim == 3 and image.shape[-1] == 1:
            image = np.repeat(image, 3, axis=-1)
        return Image.fromarray(image).convert("RGB")
    raise TypeError(f"Unsupported image type: {type(image)!r}")


class SegEarthOV3Pipeline(Pipeline):
    """Open-vocabulary semantic segmentation pipeline on native SAM 3.

    Call with a list of text prompts instead of a dataset class-name file::

        pipe = pipeline(text_prompts=["building", "road", "water"])
        result = pipe("image.tif")
        # or override prompts per call
        result = pipe("image.tif", text_prompts=["car", "tree"])
    """

    def __init__(
        self,
        model=None,
        tokenizer=None,
        feature_extractor=None,
        image_processor=None,
        processor=None,
        text_prompts: list | None = None,
        confidence_threshold: float = 0.5,
        use_sem_seg: bool = True,
        use_presence_score: bool = True,
        use_transformer_decoder: bool = True,
        slide_crop: int = 0,
        slide_stride: int = 0,
        prob_thd: float = 0.0,
        bg_idx: int = 0,
        **kwargs,
    ):
        if processor is None and image_processor is not None and tokenizer is not None:
            try:
                from transformers import Sam3Processor

                processor = Sam3Processor(image_processor, tokenizer)
            except Exception:
                processor = None

        try:
            super().__init__(
                model=model,
                tokenizer=tokenizer,
                feature_extractor=feature_extractor,
                image_processor=image_processor,
                processor=processor,
                **kwargs,
            )
        except TypeError:
            super().__init__(
                model=model,
                tokenizer=tokenizer,
                feature_extractor=feature_extractor,
                image_processor=image_processor,
                **kwargs,
            )
        self.sam3_processor = processor
        if self.sam3_processor is None:
            raise ValueError(
                "SegEarthOV3Pipeline needs a transformers Sam3Processor "
                "(or image_processor + tokenizer)."
            )
        self.default_text_prompts = text_prompts
        self.default_slide_crop = slide_crop
        self.default_slide_stride = slide_stride
        self.default_prob_thd = prob_thd
        self.default_bg_idx = bg_idx
        self.engine = Sam3OVSSEngine(
            model=self.model,
            processor=self.sam3_processor,
            confidence_threshold=confidence_threshold,
            use_sem_seg=use_sem_seg,
            use_presence_score=use_presence_score,
            use_transformer_decoder=use_transformer_decoder,
        )

    @classmethod
    def from_pretrained(cls, model_id: str = DEFAULT_MODEL_ID, **kwargs):
        """Load facebook/sam3 via transformers and wrap it in this pipeline."""
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

    def _sanitize_parameters(
        self,
        text_prompts=None,
        slide_crop=None,
        slide_stride=None,
        prob_thd=None,
        bg_idx=None,
        confidence_threshold=None,
        use_sem_seg=None,
        use_presence_score=None,
        use_transformer_decoder=None,
        **kwargs,
    ):
        preprocess_kwargs: dict[str, Any] = {}
        forward_kwargs: dict[str, Any] = {}
        postprocess_kwargs: dict[str, Any] = {}

        prompts = text_prompts if text_prompts is not None else self.default_text_prompts
        if prompts is not None:
            spec = normalize_text_prompts(prompts)
            preprocess_kwargs["prompt_spec"] = spec
            postprocess_kwargs["prompt_spec"] = spec

        preprocess_kwargs["slide_crop"] = (
            self.default_slide_crop if slide_crop is None else slide_crop
        )
        preprocess_kwargs["slide_stride"] = (
            self.default_slide_stride if slide_stride is None else slide_stride
        )
        postprocess_kwargs["prob_thd"] = (
            self.default_prob_thd if prob_thd is None else prob_thd
        )
        postprocess_kwargs["bg_idx"] = self.default_bg_idx if bg_idx is None else bg_idx

        if confidence_threshold is not None:
            forward_kwargs["confidence_threshold"] = confidence_threshold
        if use_sem_seg is not None:
            forward_kwargs["use_sem_seg"] = use_sem_seg
        if use_presence_score is not None:
            forward_kwargs["use_presence_score"] = use_presence_score
        if use_transformer_decoder is not None:
            forward_kwargs["use_transformer_decoder"] = use_transformer_decoder
        return preprocess_kwargs, forward_kwargs, postprocess_kwargs

    def preprocess(self, image, prompt_spec: PromptSpec | None = None, slide_crop=0, slide_stride=0):
        if prompt_spec is None:
            if self.default_text_prompts is None:
                raise ValueError(
                    "Pass text_prompts=['class a', 'class b', ...] to the pipeline "
                    "constructor or to the call."
                )
            prompt_spec = normalize_text_prompts(self.default_text_prompts)
        pil_image = load_any_image(image)
        return {
            "image": pil_image,
            "prompt_spec": prompt_spec,
            "slide_crop": slide_crop,
            "slide_stride": slide_stride,
            "original_size": (pil_image.size[1], pil_image.size[0]),
        }

    def _forward(self, model_inputs, **forward_kwargs):
        if "confidence_threshold" in forward_kwargs:
            self.engine.confidence_threshold = forward_kwargs["confidence_threshold"]
        if "use_sem_seg" in forward_kwargs:
            self.engine.use_sem_seg = forward_kwargs["use_sem_seg"]
        if "use_presence_score" in forward_kwargs:
            self.engine.use_presence_score = forward_kwargs["use_presence_score"]
        if "use_transformer_decoder" in forward_kwargs:
            self.engine.use_transformer_decoder = forward_kwargs["use_transformer_decoder"]

        class_logits = self.engine.infer_image(
            model_inputs["image"],
            model_inputs["prompt_spec"],
            slide_crop=model_inputs["slide_crop"],
            slide_stride=model_inputs["slide_stride"],
        )
        return {
            "class_logits": class_logits,
            "prompt_spec": model_inputs["prompt_spec"],
            "original_size": model_inputs["original_size"],
        }

    def postprocess(self, model_outputs, prompt_spec: PromptSpec | None = None, prob_thd=0.0, bg_idx=0):
        from .fusion import logits_to_prediction

        spec = prompt_spec or model_outputs["prompt_spec"]
        class_logits = model_outputs["class_logits"]
        pred = logits_to_prediction(class_logits, prob_thd=prob_thd, bg_idx=bg_idx)
        return {
            "pred_sem_seg": pred.detach().cpu().numpy().astype(np.int64),
            "seg_logits": class_logits.detach().cpu().float().numpy(),
            "text_prompts": spec.class_prompts,
            "query_words": spec.query_words,
        }


def register_pipeline() -> None:
    """Register ``segearth-ov3-segmentation`` with transformers' pipeline registry."""
    if PIPELINE_REGISTRY is None:
        return
    try:
        from transformers import Sam3Model
    except ImportError:
        return

    if "segearth-ov3-segmentation" in getattr(PIPELINE_REGISTRY, "supported_tasks", {}):
        return
    try:
        PIPELINE_REGISTRY.register_pipeline(
            "segearth-ov3-segmentation",
            pipeline_class=SegEarthOV3Pipeline,
            pt_model=Sam3Model,
            default={"pt": (DEFAULT_MODEL_ID, "main")},
            type="image",
        )
    except Exception:
        # Registry API changed or registration is unsupported in this build.
        return


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
