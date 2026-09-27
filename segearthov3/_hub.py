"""Load the Hub-standard ``hub/pipeline.py`` as the single implementation."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

HUB_DIR = Path(__file__).resolve().parents[1] / "hub"
HUB_PIPELINE_PATH = HUB_DIR / "pipeline.py"
_MODULE_NAME = "segearth_ov3_hub_pipeline"


def load_hub_pipeline_module():
    """Import ``hub/pipeline.py`` (no package-relative imports; Hub-loadable)."""
    cached = sys.modules.get(_MODULE_NAME)
    if cached is not None:
        return cached
    if not HUB_PIPELINE_PATH.is_file():
        raise FileNotFoundError(
            f"Missing Hub pipeline snapshot: {HUB_PIPELINE_PATH}. "
            "Publishable Transformers code lives in hub/pipeline.py."
        )
    spec = importlib.util.spec_from_file_location(_MODULE_NAME, HUB_PIPELINE_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[_MODULE_NAME] = module
    spec.loader.exec_module(module)
    return module
