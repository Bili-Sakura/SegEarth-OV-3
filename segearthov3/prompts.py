"""Re-export prompt helpers from the Hub-standard ``hub/pipeline.py``."""

from ._hub import load_hub_pipeline_module

_hub = load_hub_pipeline_module()

PromptSpec = _hub.PromptSpec
normalize_text_prompts = _hub.normalize_text_prompts

__all__ = ["PromptSpec", "normalize_text_prompts"]
