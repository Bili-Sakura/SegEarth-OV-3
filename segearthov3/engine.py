"""Re-export the SAM 3 engine from the Hub-standard ``hub/pipeline.py``."""

from ._hub import load_hub_pipeline_module

_hub = load_hub_pipeline_module()

Sam3OVSSEngine = _hub.Sam3OVSSEngine
infer_model_device = _hub.infer_model_device
autocast_context = _hub.autocast_context
move_to_device = _hub.move_to_device

__all__ = [
    "Sam3OVSSEngine",
    "autocast_context",
    "infer_model_device",
    "move_to_device",
]
