"""The Hub snapshot must look like a standard Transformers custom pipeline."""

from __future__ import annotations

import ast
import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HUB_DIR = ROOT / "hub"
HUB_PIPELINE = HUB_DIR / "pipeline.py"
HUB_CONFIG = HUB_DIR / "config.json"


class HubLayoutTests(unittest.TestCase):
    def test_required_hub_files_exist(self):
        self.assertTrue(HUB_PIPELINE.is_file())
        self.assertTrue(HUB_CONFIG.is_file())
        self.assertTrue((HUB_DIR / "README.md").is_file())

    def test_pipeline_has_no_package_relative_imports(self):
        tree = ast.parse(HUB_PIPELINE.read_text(), filename=str(HUB_PIPELINE))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                self.assertEqual(
                    node.level,
                    0,
                    msg="hub/pipeline.py must not use relative imports",
                )
                module = node.module or ""
                self.assertFalse(
                    module.startswith("segearthov3"),
                    msg="hub/pipeline.py must not import the GitHub package",
                )
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    self.assertFalse(
                        alias.name.startswith("segearthov3"),
                        msg="hub/pipeline.py must not import the GitHub package",
                    )

    def test_config_registers_official_custom_pipelines(self):
        config = json.loads(HUB_CONFIG.read_text())
        self.assertEqual(config["model_type"], "sam3")
        self.assertEqual(config["architectures"], ["Sam3Model"])
        task = config["custom_pipelines"]["segearth-ov3-segmentation"]
        self.assertEqual(task["impl"], "pipeline.SegEarthOV3Pipeline")
        self.assertEqual(task["pt"], ["AutoModel"])
        self.assertEqual(task["type"], "image")
        self.assertIsInstance(config["text_prompts"], list)
        self.assertGreaterEqual(len(config["text_prompts"]), 2)

    def test_sam3_config_roundtrip_keeps_custom_pipelines(self):
        from transformers import Sam3Config

        raw = json.loads(HUB_CONFIG.read_text())
        loaded = Sam3Config.from_dict(raw)
        self.assertEqual(loaded.model_type, "sam3")
        self.assertEqual(
            loaded.custom_pipelines["segearth-ov3-segmentation"]["impl"],
            "pipeline.SegEarthOV3Pipeline",
        )
        self.assertEqual(loaded.text_prompts, raw["text_prompts"])

    def test_package_reexports_the_hub_implementation(self):
        from segearthov3._hub import load_hub_pipeline_module
        from segearthov3.engine import Sam3OVSSEngine
        from segearthov3.fusion import fuse_sam3_output
        from segearthov3.pipeline import SegEarthOV3Pipeline
        from segearthov3.prompts import PromptSpec, normalize_text_prompts

        hub = load_hub_pipeline_module()
        self.assertIs(PromptSpec, hub.PromptSpec)
        self.assertIs(normalize_text_prompts, hub.normalize_text_prompts)
        self.assertIs(fuse_sam3_output, hub.fuse_sam3_output)
        self.assertIs(Sam3OVSSEngine, hub.Sam3OVSSEngine)
        self.assertIs(SegEarthOV3Pipeline, hub.SegEarthOV3Pipeline)

    def test_registry_uses_automodel(self):
        from transformers import AutoModel
        from transformers.pipelines import PIPELINE_REGISTRY

        from segearthov3.pipeline import register_pipeline

        from segearthov3.pipeline import SegEarthOV3Pipeline

        register_pipeline()
        task = PIPELINE_REGISTRY.supported_tasks["segearth-ov3-segmentation"]
        self.assertIn(AutoModel, task["pt"])
        self.assertIs(task["impl"], SegEarthOV3Pipeline)


if __name__ == "__main__":
    unittest.main()
