"""Fuse SAM 3 instance, semantic, and presence heads into class logits."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F


def resize_map(tensor: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    """Bilinear-resize a 2D or batched 2D map to ``size=(H, W)``."""
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
            tensor.unsqueeze(1),
            size=size,
            mode="bilinear",
            align_corners=False,
        ).squeeze(1)
    if tensor.ndim == 4:
        return F.interpolate(tensor, size=size, mode="bilinear", align_corners=False)
    raise ValueError(f"Unsupported tensor rank {tensor.ndim} for resize")


def _as_presence_score(presence_logits: torch.Tensor | None) -> torch.Tensor | None:
    if presence_logits is None:
        return None
    score = presence_logits.sigmoid().reshape(-1)[0]
    return score


def fuse_sam3_output(
    outputs: Any,
    target_size: tuple[int, int],
    confidence_threshold: float = 0.5,
    use_sem_seg: bool = True,
    use_presence_score: bool = True,
    use_transformer_decoder: bool = True,
) -> torch.Tensor:
    """Fuse one-prompt SAM 3 outputs into a single ``[H, W]`` logit map.

    Mirrors the SegEarth-OV3 paper: instance aggregation, dual-head max fusion,
    then presence-guided scaling.
    """
    height, width = target_size
    pred_masks = outputs.pred_masks
    if pred_masks is None:
        raise ValueError("SAM 3 outputs are missing pred_masks")

    device = pred_masks.device
    dtype = pred_masks.dtype
    fused = torch.zeros((height, width), device=device, dtype=dtype)

    # Drop the batch dimension used by transformers Sam3Model.
    if pred_masks.ndim == 4:
        pred_masks = pred_masks[0]
    pred_logits = getattr(outputs, "pred_logits", None)
    if pred_logits is not None and pred_logits.ndim == 2:
        pred_logits = pred_logits[0]

    presence_score = _as_presence_score(getattr(outputs, "presence_logits", None))

    if use_transformer_decoder and pred_logits is not None and pred_masks.numel() > 0:
        object_scores = pred_logits.sigmoid()
        if presence_score is not None:
            object_scores = object_scores * presence_score
        keep = object_scores > confidence_threshold
        if keep.any():
            instance_logits = pred_masks[keep].sigmoid()
            instance_logits = resize_map(instance_logits, (height, width))
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
            semantic = resize_map(semantic.sigmoid(), (height, width))
            fused = torch.maximum(fused, semantic.to(dtype=fused.dtype))

    if use_presence_score and presence_score is not None:
        fused = fused * presence_score.to(dtype=fused.dtype)

    return fused


def aggregate_class_logits(
    query_logits: torch.Tensor,
    query_idx: torch.Tensor,
    num_classes: int,
) -> torch.Tensor:
    """Max-pool synonym queries that belong to the same class.

    Args:
        query_logits: ``[num_queries, H, W]``
        query_idx: ``[num_queries]`` int class ids
        num_classes: number of semantic classes
    """
    if query_logits.shape[0] == num_classes and torch.equal(
        query_idx, torch.arange(num_classes, device=query_idx.device)
    ):
        return query_logits

    class_index = F.one_hot(query_idx.long(), num_classes=num_classes).T
    class_index = class_index.view(num_classes, query_logits.shape[0], 1, 1).to(
        dtype=query_logits.dtype
    )
    return (query_logits.unsqueeze(0) * class_index).max(dim=1).values


def logits_to_prediction(
    class_logits: torch.Tensor,
    prob_thd: float = 0.0,
    bg_idx: int = 0,
) -> torch.Tensor:
    """Argmax class map with a low-confidence fallback to ``bg_idx``."""
    pred = class_logits.argmax(dim=0)
    if prob_thd > 0:
        pred = pred.clone()
        pred[class_logits.max(dim=0).values < prob_thd] = bg_idx
    return pred
