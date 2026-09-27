"""Native transformers SAM 3 inference engine used by the custom pipeline."""

from __future__ import annotations

from contextlib import nullcontext
from typing import Any

import torch
from PIL import Image

from .fusion import (
    aggregate_class_logits,
    fuse_sam3_output,
    logits_to_prediction,
    resize_map,
)
from .prompts import PromptSpec, normalize_text_prompts


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
    if torch.is_tensor(value):
        return value.to(device)
    return value


class Sam3OVSSEngine:
    """Run SegEarth-OV3 fusion on top of ``transformers.Sam3Model``."""

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
        if self._device is not None:
            return self._device
        return infer_model_device(self.model)

    def encode_image(self, image: Image.Image):
        """Precompute vision embeddings once for many text prompts."""
        device = self.device
        img_inputs = self.processor(images=image, return_tensors="pt")
        pixel_values = move_to_device(img_inputs["pixel_values"], device)
        with torch.inference_mode(), autocast_context(device):
            vision_embeds = self.model.get_vision_features(pixel_values=pixel_values)
        return vision_embeds, img_inputs

    def infer_prompt(
        self,
        vision_embeds,
        prompt: str,
        target_size: tuple[int, int],
    ) -> torch.Tensor:
        device = self.device
        text_inputs = self.processor(text=prompt, return_tensors="pt")
        model_kwargs = {
            "input_ids": move_to_device(text_inputs["input_ids"], device),
        }
        if "attention_mask" in text_inputs and text_inputs["attention_mask"] is not None:
            model_kwargs["attention_mask"] = move_to_device(
                text_inputs["attention_mask"], device
            )

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

    def infer_single_view(
        self,
        image: Image.Image,
        query_words: list[str],
    ) -> torch.Tensor:
        width, height = image.size
        vision_embeds, _ = self.encode_image(image)
        query_maps = [
            self.infer_prompt(vision_embeds, prompt, (height, width))
            for prompt in query_words
        ]
        return torch.stack(query_maps, dim=0)

    def slide_inference(
        self,
        image: Image.Image,
        query_words: list[str],
        stride: int | tuple[int, int],
        crop_size: int | tuple[int, int],
    ) -> torch.Tensor:
        width, height = image.size
        if isinstance(stride, int):
            stride = (stride, stride)
        if isinstance(crop_size, int):
            crop_size = (crop_size, crop_size)

        h_stride, w_stride = stride
        h_crop, w_crop = crop_size
        device = self.device
        preds = torch.zeros((len(query_words), height, width), device=device)
        count = torch.zeros((1, height, width), device=device)

        h_grids = max(height - h_crop + h_stride - 1, 0) // h_stride + 1
        w_grids = max(width - w_crop + w_stride - 1, 0) // w_stride + 1

        for h_idx in range(h_grids):
            for w_idx in range(w_grids):
                y1 = h_idx * h_stride
                x1 = w_idx * w_stride
                y2 = min(y1 + h_crop, height)
                x2 = min(x1 + w_crop, width)
                y1 = max(y2 - h_crop, 0)
                x1 = max(x2 - w_crop, 0)
                crop = image.crop((x1, y1, x2, y2))
                crop_logits = self.infer_single_view(crop, query_words)
                preds[:, y1:y2, x1:x2] += crop_logits
                count[:, y1:y2, x1:x2] += 1

        if (count == 0).any():
            raise RuntimeError("Sliding window did not cover the full image")
        return preds / count

    def infer_image(
        self,
        image: Image.Image,
        text_prompts: list[str] | PromptSpec,
        slide_crop: int = 0,
        slide_stride: int = 0,
    ) -> torch.Tensor:
        spec = (
            text_prompts
            if isinstance(text_prompts, PromptSpec)
            else normalize_text_prompts(text_prompts)
        )
        width, height = image.size
        if slide_crop > 0 and (slide_crop < width or slide_crop < height):
            query_logits = self.slide_inference(
                image, spec.query_words, slide_stride, slide_crop
            )
        else:
            query_logits = self.infer_single_view(image, spec.query_words)

        query_idx = torch.tensor(spec.query_idx, device=query_logits.device, dtype=torch.long)
        return aggregate_class_logits(query_logits, query_idx, spec.num_classes)

    def predict_mask(
        self,
        image: Image.Image,
        text_prompts: list[str] | PromptSpec,
        slide_crop: int = 0,
        slide_stride: int = 0,
        prob_thd: float = 0.0,
        bg_idx: int = 0,
        target_size: tuple[int, int] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        class_logits = self.infer_image(
            image,
            text_prompts,
            slide_crop=slide_crop,
            slide_stride=slide_stride,
        )
        if target_size is not None and tuple(class_logits.shape[-2:]) != target_size:
            class_logits = resize_map(class_logits, target_size)
        pred = logits_to_prediction(class_logits, prob_thd=prob_thd, bg_idx=bg_idx)
        return class_logits, pred

    def extract_fpn_features(self, image: Image.Image):
        vision_embeds, _ = self.encode_image(image)
        fpn = getattr(vision_embeds, "fpn_hidden_states", None)
        if fpn is None:
            raise ValueError("transformers Sam3 vision output has no fpn_hidden_states")
        return fpn, (image.size[1], image.size[0])
