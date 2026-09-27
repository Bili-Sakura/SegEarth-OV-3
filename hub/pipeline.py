"""SegEarth-OV3 custom pipeline for a Hugging Face model repo.

This file is Hub-loadable. It has no package-relative imports so transformers
can fetch it with ``trust_remote_code=True``:

    from transformers import pipeline
    pipe = pipeline(
        "segearth-ov3-segmentation",
        model="YOUR_NAMESPACE/SegEarth-OV-3",
        trust_remote_code=True,
        text_prompts=["building", "road", "water"],
    )

The task is declared in this repo's ``config.json`` under ``custom_pipelines``.
Weights come from native ``transformers.Sam3Model`` (typically ``facebook/sam3``).
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from transformers import Pipeline

DEFAULT_MODEL_ID = "facebook/sam3"


@dataclass(frozen=True)
class PromptSpec:
    query_words: list[str]
    query_idx: list[int]
    class_prompts: list[str]

    @property
    def num_queries(self) -> int:
        return len(self.query_words)

    @property
    def num_classes(self) -> int:
        return len(self.class_prompts)


def _split_synonyms(item: str | Sequence[str]) -> list[str]:
    if isinstance(item, str):
        parts = [part.strip() for part in item.split(",")]
    elif isinstance(item, Sequence):
        parts = []
        for nested in item:
            if not isinstance(nested, str):
                raise TypeError(
                    f"Prompt synonym groups must contain strings, got {type(nested)!r}"
                )
            parts.extend(_split_synonyms(nested))
    else:
        raise TypeError(
            "Each text prompt must be a string or a sequence of synonym strings, "
            f"got {type(item)!r}"
        )
    names = [part.replace("\n", "").strip() for part in parts if part.strip()]
    if not names:
        raise ValueError("Encountered an empty text prompt after stripping whitespace")
    return names


def normalize_text_prompts(text_prompts: Iterable[str | Sequence[str]]) -> PromptSpec:
    """Convert a prompt list into per-query strings and class indices.

    Accepted forms: ``["building", "road"]``, ``["building,house"]``,
    or ``[["building", "house"], "road"]``.
    """
    if text_prompts is None:
        raise ValueError("text_prompts is required and must be a non-empty list")
    if isinstance(text_prompts, str):
        raise TypeError(
            "text_prompts must be a list of prompts, not a single string. "
            "Use ['building', 'road'] instead of 'building'."
        )
    items = list(text_prompts)
    if not items:
        raise ValueError("text_prompts must contain at least one class prompt")

    query_words: list[str] = []
    query_idx: list[int] = []
    class_prompts: list[str] = []
    for class_id, item in enumerate(items):
        synonyms = _split_synonyms(item)
        class_prompts.append(",".join(synonyms))
        query_words.extend(synonyms)
        query_idx.extend([class_id] * len(synonyms))
    return PromptSpec(query_words=query_words, query_idx=query_idx, class_prompts=class_prompts)


def load_any_image(image: Any) -> Image.Image:
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


def resize_map(tensor: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    if tuple(tensor.shape[-2:]) == size:
        return tensor
    if tensor.ndim == 2:
        return F.interpolate(
            tensor.view(1, 1, *tensor.shape),
            size=size,
            mode="bilinear",
            align_corners=False,
        ).squeeze(0).squeeze(0)
    if tensor.ndim == 3:
        return F.interpolate(
            tensor.unsqueeze(1), size=size, mode="bilinear", align_corners=False
        ).squeeze(1)
    if tensor.ndim == 4:
        return F.interpolate(tensor, size=size, mode="bilinear", align_corners=False)
    raise ValueError(f"Unsupported tensor rank {tensor.ndim} for resize")


def fuse_sam3_output(
    outputs: Any,
    target_size: tuple[int, int],
    confidence_threshold: float = 0.5,
    use_sem_seg: bool = True,
    use_presence_score: bool = True,
    use_transformer_decoder: bool = True,
) -> torch.Tensor:
    height, width = target_size
    pred_masks = outputs.pred_masks
    if pred_masks is None:
        raise ValueError("SAM 3 outputs are missing pred_masks")

    fused = torch.zeros((height, width), device=pred_masks.device, dtype=pred_masks.dtype)
    if pred_masks.ndim == 4:
        pred_masks = pred_masks[0]
    pred_logits = getattr(outputs, "pred_logits", None)
    if pred_logits is not None and pred_logits.ndim == 2:
        pred_logits = pred_logits[0]

    presence = getattr(outputs, "presence_logits", None)
    presence_score = None if presence is None else presence.sigmoid().reshape(-1)[0]

    if use_transformer_decoder and pred_logits is not None and pred_masks.numel() > 0:
        object_scores = pred_logits.sigmoid()
        if presence_score is not None:
            object_scores = object_scores * presence_score
        keep = object_scores > confidence_threshold
        if keep.any():
            instance_logits = resize_map(pred_masks[keep].sigmoid(), (height, width))
            instance_scores = object_scores[keep].to(dtype=instance_logits.dtype)
            while instance_scores.ndim < instance_logits.ndim:
                instance_scores = instance_scores.unsqueeze(-1)
            fused = torch.maximum(fused, (instance_logits * instance_scores).max(dim=0).values)

    if use_sem_seg:
        semantic = getattr(outputs, "semantic_seg", None)
        if semantic is not None:
            if semantic.ndim == 4:
                semantic = semantic[0, 0]
            elif semantic.ndim == 3:
                semantic = semantic[0]
            fused = torch.maximum(fused, resize_map(semantic.sigmoid(), (height, width)).to(fused.dtype))

    if use_presence_score and presence_score is not None:
        fused = fused * presence_score.to(dtype=fused.dtype)
    return fused


def aggregate_class_logits(
    query_logits: torch.Tensor, query_idx: torch.Tensor, num_classes: int
) -> torch.Tensor:
    if query_logits.shape[0] == num_classes and torch.equal(
        query_idx, torch.arange(num_classes, device=query_idx.device)
    ):
        return query_logits
    class_index = F.one_hot(query_idx.long(), num_classes=num_classes).T
    class_index = class_index.view(num_classes, query_logits.shape[0], 1, 1).to(dtype=query_logits.dtype)
    return (query_logits.unsqueeze(0) * class_index).max(dim=1).values


def logits_to_prediction(class_logits: torch.Tensor, prob_thd: float = 0.0, bg_idx: int = 0) -> torch.Tensor:
    pred = class_logits.argmax(dim=0)
    if prob_thd > 0:
        pred = pred.clone()
        pred[class_logits.max(dim=0).values < prob_thd] = bg_idx
    return pred


def infer_model_device(model: torch.nn.Module) -> torch.device:
    try:
        device = getattr(model, "device", None)
        if isinstance(device, torch.device):
            return device
    except Exception:
        pass
    return next(model.parameters()).device


def autocast_context(device: torch.device):
    if device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def move_to_device(value: Any, device: torch.device) -> Any:
    return value.to(device) if torch.is_tensor(value) else value


class Sam3OVSSEngine:
    """Cache vision embeddings and fuse SAM 3 heads for a list of text prompts."""

    def __init__(
        self,
        model,
        processor,
        confidence_threshold: float = 0.5,
        use_sem_seg: bool = True,
        use_presence_score: bool = True,
        use_transformer_decoder: bool = True,
        device: torch.device | str | None = None,
    ):
        self.model = model
        self.processor = processor
        self.confidence_threshold = confidence_threshold
        self.use_sem_seg = use_sem_seg
        self.use_presence_score = use_presence_score
        self.use_transformer_decoder = use_transformer_decoder
        self._device = torch.device(device) if device is not None else None
        self.model.eval()

    @property
    def device(self) -> torch.device:
        return self._device if self._device is not None else infer_model_device(self.model)

    def encode_image(self, image: Image.Image):
        device = self.device
        img_inputs = self.processor(images=image, return_tensors="pt")
        pixel_values = move_to_device(img_inputs["pixel_values"], device)
        with torch.inference_mode(), autocast_context(device):
            vision_embeds = self.model.get_vision_features(pixel_values=pixel_values)
        return vision_embeds, img_inputs

    def infer_prompt(self, vision_embeds, prompt: str, target_size: tuple[int, int]) -> torch.Tensor:
        device = self.device
        text_inputs = self.processor(text=prompt, return_tensors="pt")
        model_kwargs = {"input_ids": move_to_device(text_inputs["input_ids"], device)}
        if text_inputs.get("attention_mask") is not None:
            model_kwargs["attention_mask"] = move_to_device(text_inputs["attention_mask"], device)
        with torch.inference_mode(), autocast_context(device):
            outputs = self.model(vision_embeds=vision_embeds, **model_kwargs)
            fused = fuse_sam3_output(
                outputs,
                target_size=target_size,
                confidence_threshold=self.confidence_threshold,
                use_sem_seg=self.use_sem_seg,
                use_presence_score=self.use_presence_score,
                use_transformer_decoder=self.use_transformer_decoder,
            )
        return fused.float()

    def infer_single_view(self, image: Image.Image, query_words: list[str]) -> torch.Tensor:
        width, height = image.size
        vision_embeds, _ = self.encode_image(image)
        return torch.stack(
            [self.infer_prompt(vision_embeds, prompt, (height, width)) for prompt in query_words],
            dim=0,
        )

    def slide_inference(self, image: Image.Image, query_words: list[str], stride, crop_size) -> torch.Tensor:
        width, height = image.size
        if isinstance(stride, int):
            stride = (stride, stride)
        if isinstance(crop_size, int):
            crop_size = (crop_size, crop_size)
        h_stride, w_stride = stride
        h_crop, w_crop = crop_size
        preds = torch.zeros((len(query_words), height, width), device=self.device)
        count = torch.zeros((1, height, width), device=self.device)
        h_grids = max(height - h_crop + h_stride - 1, 0) // h_stride + 1
        w_grids = max(width - w_crop + w_stride - 1, 0) // w_stride + 1
        for h_idx in range(h_grids):
            for w_idx in range(w_grids):
                y1, x1 = h_idx * h_stride, w_idx * w_stride
                y2, x2 = min(y1 + h_crop, height), min(x1 + w_crop, width)
                y1, x1 = max(y2 - h_crop, 0), max(x2 - w_crop, 0)
                preds[:, y1:y2, x1:x2] += self.infer_single_view(image.crop((x1, y1, x2, y2)), query_words)
                count[:, y1:y2, x1:x2] += 1
        if (count == 0).any():
            raise RuntimeError("Sliding window did not cover the full image")
        return preds / count

    def infer_image(
        self,
        image: Image.Image,
        text_prompts: list | PromptSpec,
        slide_crop: int = 0,
        slide_stride: int = 0,
    ) -> torch.Tensor:
        spec = text_prompts if isinstance(text_prompts, PromptSpec) else normalize_text_prompts(text_prompts)
        width, height = image.size
        if slide_crop > 0 and (slide_crop < width or slide_crop < height):
            query_logits = self.slide_inference(image, spec.query_words, slide_stride, slide_crop)
        else:
            query_logits = self.infer_single_view(image, spec.query_words)
        query_idx = torch.tensor(spec.query_idx, device=query_logits.device, dtype=torch.long)
        return aggregate_class_logits(query_logits, query_idx, spec.num_classes)

    def predict_mask(
        self,
        image: Image.Image,
        text_prompts: list | PromptSpec,
        slide_crop: int = 0,
        slide_stride: int = 0,
        prob_thd: float = 0.0,
        bg_idx: int = 0,
        target_size: tuple[int, int] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        class_logits = self.infer_image(
            image, text_prompts, slide_crop=slide_crop, slide_stride=slide_stride
        )
        if target_size is not None and tuple(class_logits.shape[-2:]) != target_size:
            class_logits = resize_map(class_logits, target_size)
        return class_logits, logits_to_prediction(class_logits, prob_thd=prob_thd, bg_idx=bg_idx)

    def extract_fpn_features(self, image: Image.Image):
        vision_embeds, _ = self.encode_image(image)
        fpn = getattr(vision_embeds, "fpn_hidden_states", None)
        if fpn is None:
            raise ValueError("transformers Sam3 vision output has no fpn_hidden_states")
        return fpn, (image.size[1], image.size[0])


def _config_value(model, key: str, default):
    config = getattr(model, "config", None)
    if config is None:
        return default
    return getattr(config, key, default)


class SegEarthOV3Pipeline(Pipeline):
    """Open-vocabulary remote-sensing segmentation on native SAM 3.

    ``text_prompts`` is a list of class names (or synonym groups). Defaults are
    read from ``model.config.text_prompts`` when the model was loaded from this
    Hub repo's ``config.json``.
    """

    def __init__(self, *args, **kwargs):
        text_prompts = kwargs.pop("text_prompts", None)
        confidence_threshold = kwargs.pop("confidence_threshold", None)
        use_sem_seg = kwargs.pop("use_sem_seg", None)
        use_presence_score = kwargs.pop("use_presence_score", None)
        use_transformer_decoder = kwargs.pop("use_transformer_decoder", None)
        slide_crop = kwargs.pop("slide_crop", None)
        slide_stride = kwargs.pop("slide_stride", None)
        prob_thd = kwargs.pop("prob_thd", None)
        bg_idx = kwargs.pop("bg_idx", None)
        processor = kwargs.get("processor")

        super().__init__(*args, **kwargs)

        if processor is None and self.image_processor is not None and self.tokenizer is not None:
            from transformers import Sam3Processor

            processor = Sam3Processor(self.image_processor, self.tokenizer)
        self.sam3_processor = processor or getattr(self, "processor", None)
        if self.sam3_processor is None:
            raise ValueError(
                "SegEarthOV3Pipeline needs a Sam3Processor (or image_processor + tokenizer)."
            )

        self.default_text_prompts = (
            text_prompts if text_prompts is not None else _config_value(self.model, "text_prompts", None)
        )
        self.default_slide_crop = (
            slide_crop if slide_crop is not None else int(_config_value(self.model, "slide_crop", 0))
        )
        self.default_slide_stride = (
            slide_stride if slide_stride is not None else int(_config_value(self.model, "slide_stride", 0))
        )
        self.default_prob_thd = (
            prob_thd if prob_thd is not None else float(_config_value(self.model, "prob_thd", 0.0))
        )
        self.default_bg_idx = bg_idx if bg_idx is not None else int(_config_value(self.model, "bg_idx", 0))
        self.engine = Sam3OVSSEngine(
            model=self.model,
            processor=self.sam3_processor,
            confidence_threshold=(
                confidence_threshold
                if confidence_threshold is not None
                else float(_config_value(self.model, "confidence_threshold", 0.5))
            ),
            use_sem_seg=use_sem_seg if use_sem_seg is not None else bool(_config_value(self.model, "use_sem_seg", True)),
            use_presence_score=(
                use_presence_score
                if use_presence_score is not None
                else bool(_config_value(self.model, "use_presence_score", True))
            ),
            use_transformer_decoder=(
                use_transformer_decoder
                if use_transformer_decoder is not None
                else bool(_config_value(self.model, "use_transformer_decoder", True))
            ),
        )

    def _sanitize_parameters(self, **kwargs):
        preprocess_kwargs: dict[str, Any] = {}
        forward_kwargs: dict[str, Any] = {}
        postprocess_kwargs: dict[str, Any] = {}

        prompts = kwargs.get("text_prompts", self.default_text_prompts)
        if prompts is not None:
            spec = normalize_text_prompts(prompts)
            preprocess_kwargs["prompt_spec"] = spec
            postprocess_kwargs["prompt_spec"] = spec

        if "slide_crop" in kwargs or self.default_slide_crop:
            preprocess_kwargs["slide_crop"] = kwargs.get("slide_crop", self.default_slide_crop)
        if "slide_stride" in kwargs or self.default_slide_stride:
            preprocess_kwargs["slide_stride"] = kwargs.get("slide_stride", self.default_slide_stride)
        if "prob_thd" in kwargs or self.default_prob_thd:
            postprocess_kwargs["prob_thd"] = kwargs.get("prob_thd", self.default_prob_thd)
        if "bg_idx" in kwargs or self.default_bg_idx:
            postprocess_kwargs["bg_idx"] = kwargs.get("bg_idx", self.default_bg_idx)

        for key in (
            "confidence_threshold",
            "use_sem_seg",
            "use_presence_score",
            "use_transformer_decoder",
        ):
            if key in kwargs:
                forward_kwargs[key] = kwargs[key]
        return preprocess_kwargs, forward_kwargs, postprocess_kwargs

    def preprocess(self, image, prompt_spec: PromptSpec | None = None, slide_crop=0, slide_stride=0):
        if prompt_spec is None:
            if self.default_text_prompts is None:
                raise ValueError(
                    "Pass text_prompts=['class a', 'class b', ...] or set them in config.json."
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
        for key in (
            "confidence_threshold",
            "use_sem_seg",
            "use_presence_score",
            "use_transformer_decoder",
        ):
            if key in forward_kwargs:
                setattr(self.engine, key, forward_kwargs[key])
        class_logits = self.engine.infer_image(
            model_inputs["image"],
            model_inputs["prompt_spec"],
            slide_crop=model_inputs.get("slide_crop", 0),
            slide_stride=model_inputs.get("slide_stride", 0),
        )
        return {
            "class_logits": class_logits,
            "prompt_spec": model_inputs["prompt_spec"],
            "original_size": model_inputs["original_size"],
        }

    def postprocess(self, model_outputs, prompt_spec: PromptSpec | None = None, prob_thd=0.0, bg_idx=0):
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
    """Register this file's pipeline locally (also used before push_to_hub)."""
    from transformers import AutoModel
    from transformers.pipelines import PIPELINE_REGISTRY

    if "segearth-ov3-segmentation" in getattr(PIPELINE_REGISTRY, "supported_tasks", {}):
        return
    PIPELINE_REGISTRY.register_pipeline(
        "segearth-ov3-segmentation",
        pipeline_class=SegEarthOV3Pipeline,
        pt_model=AutoModel,
        default={"pt": (DEFAULT_MODEL_ID, "main")},
        type="image",
    )
