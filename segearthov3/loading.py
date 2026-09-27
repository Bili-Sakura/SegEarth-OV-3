"""Load facebook/sam3 through transformers only (no sam3 package)."""

from __future__ import annotations

import torch


DEFAULT_MODEL_ID = "facebook/sam3"


def load_sam3(model_id: str = DEFAULT_MODEL_ID, device: str | torch.device | None = None, **kwargs):
    """Load ``Sam3Model`` and ``Sam3Processor`` from the Hugging Face Hub."""
    try:
        from transformers import Sam3Model, Sam3Processor
    except ImportError as exc:
        raise ImportError(
            "SegEarth-OV3 now loads SAM 3 via Hugging Face transformers. "
            "Install a recent transformers build that includes Sam3Model "
            "(``pip install -U transformers``)."
        ) from exc

    load_kwargs = dict(kwargs)
    resolved_device = device
    if resolved_device is None:
        resolved_device = "cuda" if torch.cuda.is_available() else "cpu"

    if (
        resolved_device != "cpu"
        and "device_map" not in load_kwargs
        and str(resolved_device) != "cpu"
    ):
        load_kwargs.setdefault("device_map", resolved_device)
        if torch.cuda.is_available():
            load_kwargs.setdefault("dtype", torch.bfloat16)

    model = Sam3Model.from_pretrained(model_id, **load_kwargs)
    processor = Sam3Processor.from_pretrained(model_id)

    if "device_map" not in load_kwargs:
        model = model.to(resolved_device)
    model.eval()
    return model, processor
