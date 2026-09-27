"""Re-export SAM 3 fusion helpers from the Hub-standard ``hub/pipeline.py``."""

from ._hub import load_hub_pipeline_module

_hub = load_hub_pipeline_module()

resize_map = _hub.resize_map
fuse_sam3_output = _hub.fuse_sam3_output
aggregate_class_logits = _hub.aggregate_class_logits
logits_to_prediction = _hub.logits_to_prediction

__all__ = [
    "aggregate_class_logits",
    "fuse_sam3_output",
    "logits_to_prediction",
    "resize_map",
]
