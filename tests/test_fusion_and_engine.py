import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import torch
from PIL import Image

from segearthov3.engine import Sam3OVSSEngine
from segearthov3.fusion import (
    aggregate_class_logits,
    fuse_sam3_output,
    logits_to_prediction,
    resize_map,
)
from segearthov3.pipeline import SegEarthOV3Pipeline, load_any_image
from segearthov3.prompts import normalize_text_prompts


class FusionTests(unittest.TestCase):
    def test_resize_map_2d(self):
        src = torch.ones(4, 4)
        out = resize_map(src, (8, 8))
        self.assertEqual(tuple(out.shape), (8, 8))

    def test_instance_semantic_presence_fusion(self):
        pred_masks = torch.full((1, 2, 4, 4), -4.0)
        pred_masks[0, 0, 1:3, 1:3] = 4.0
        pred_logits = torch.tensor([[4.0, -4.0]])
        presence = torch.tensor([[4.0]])
        semantic = torch.zeros(1, 1, 4, 4)
        semantic[0, 0, 0, 0] = 4.0
        outputs = SimpleNamespace(
            pred_masks=pred_masks,
            pred_logits=pred_logits,
            presence_logits=presence,
            semantic_seg=semantic,
        )
        fused = fuse_sam3_output(outputs, (4, 4), confidence_threshold=0.5)
        self.assertEqual(tuple(fused.shape), (4, 4))
        self.assertGreater(fused[1, 1].item(), fused[3, 3].item())
        self.assertGreater(fused[0, 0].item(), 0.0)

    def test_presence_can_suppress_absent_class(self):
        pred_masks = torch.ones(1, 1, 2, 2)
        outputs = SimpleNamespace(
            pred_masks=pred_masks,
            pred_logits=torch.tensor([[8.0]]),
            presence_logits=torch.tensor([[-20.0]]),
            semantic_seg=torch.ones(1, 1, 2, 2),
        )
        fused = fuse_sam3_output(outputs, (2, 2), confidence_threshold=0.5)
        self.assertLess(fused.max().item(), 1e-4)

    def test_aggregate_synonym_queries(self):
        query_logits = torch.stack(
            [
                torch.full((2, 2), 0.2),
                torch.full((2, 2), 0.9),
                torch.full((2, 2), 0.3),
            ]
        )
        query_idx = torch.tensor([0, 0, 1])
        class_logits = aggregate_class_logits(query_logits, query_idx, num_classes=2)
        self.assertEqual(tuple(class_logits.shape), (2, 2, 2))
        self.assertTrue(torch.allclose(class_logits[0], torch.full((2, 2), 0.9)))
        self.assertTrue(torch.allclose(class_logits[1], torch.full((2, 2), 0.3)))

    def test_probability_threshold_falls_back_to_background(self):
        logits = torch.tensor(
            [
                [[0.01, 0.90], [0.01, 0.01]],
                [[0.02, 0.10], [0.80, 0.02]],
            ]
        )
        pred = logits_to_prediction(logits, prob_thd=0.5, bg_idx=0)
        self.assertEqual(pred.tolist(), [[0, 0], [1, 0]])


class EngineAndPipelineTests(unittest.TestCase):
    def _fake_outputs(self, prompt: str, height: int, width: int):
        score = 6.0 if prompt == "building" else -6.0
        masks = torch.full((1, 1, height, width), score)
        semantic = torch.full((1, 1, height, width), score)
        return SimpleNamespace(
            pred_masks=masks,
            pred_logits=torch.tensor([[score]]),
            presence_logits=torch.tensor([[score]]),
            semantic_seg=semantic,
        )

    def _build_engine(self):
        model = MagicMock()
        model.parameters.return_value = iter([torch.zeros(1)])
        model.device = torch.device("cpu")

        def get_vision_features(pixel_values):
            return SimpleNamespace(
                fpn_hidden_states=(
                    torch.randn(1, 32, 4, 4),
                    torch.randn(1, 64, 2, 2),
                )
            )

        def forward(*, vision_embeds, input_ids, attention_mask=None):
            token = int(input_ids.reshape(-1)[0].item())
            prompt = "building" if token == 1 else "water"
            return self._fake_outputs(prompt, 8, 8)

        model.get_vision_features.side_effect = get_vision_features
        model.side_effect = forward

        processor = MagicMock()

        def process(*, images=None, text=None, return_tensors="pt"):
            if images is not None:
                return {"pixel_values": torch.zeros(1, 3, 8, 8)}
            token = 1 if text == "building" else 2
            return {
                "input_ids": torch.tensor([[token]]),
                "attention_mask": torch.ones(1, 1, dtype=torch.long),
            }

        processor.side_effect = process
        return Sam3OVSSEngine(model, processor, confidence_threshold=0.1, device="cpu")

    def test_engine_maxpools_prompts_to_classes(self):
        engine = self._build_engine()
        image = Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8))
        logits = engine.infer_image(image, ["building", "water"])
        self.assertEqual(tuple(logits.shape), (2, 8, 8))
        pred = logits.argmax(0)
        self.assertTrue((pred == 0).all())

    def test_sliding_window_covers_image(self):
        engine = self._build_engine()
        image = Image.fromarray(np.zeros((10, 12, 3), dtype=np.uint8))
        logits = engine.infer_image(
            image, ["building"], slide_crop=8, slide_stride=6
        )
        self.assertEqual(tuple(logits.shape), (1, 10, 12))
        self.assertFalse(torch.isnan(logits).any())

    def test_pipeline_call_accepts_prompt_list(self):
        engine = self._build_engine()
        pipe = SegEarthOV3Pipeline.__new__(SegEarthOV3Pipeline)
        pipe.default_text_prompts = ["building", "water"]
        pipe.default_slide_crop = 0
        pipe.default_slide_stride = 0
        pipe.default_prob_thd = 0.0
        pipe.default_bg_idx = 0
        pipe.engine = engine
        pipe.sam3_processor = engine.processor

        image = Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8))
        preprocess = pipe.preprocess(image, prompt_spec=normalize_text_prompts(["building", "water"]))
        outputs = pipe._forward(preprocess)
        result = pipe.postprocess(outputs, prompt_spec=preprocess["prompt_spec"])
        self.assertEqual(result["pred_sem_seg"].shape, (8, 8))
        self.assertEqual(result["text_prompts"], ["building", "water"])
        self.assertEqual(result["query_words"], ["building", "water"])

    def test_load_any_image_from_array(self):
        image = load_any_image(np.zeros((6, 7, 3), dtype=np.uint8))
        self.assertEqual(image.size, (7, 6))
        self.assertEqual(image.mode, "RGB")


class ConfigPromptTests(unittest.TestCase):
    def test_configs_use_text_prompts_not_txt_files(self):
        from pathlib import Path

        configs = Path("/workspace/configs")
        leftovers = []
        for cfg in configs.glob("cfg_*.py"):
            text = cfg.read_text()
            if "classname_path" in text:
                leftovers.append(cfg.name)
            if "text_prompts=" not in text and cfg.name != "base_config.py":
                leftovers.append(cfg.name)
        self.assertEqual(leftovers, [])


if __name__ == "__main__":
    unittest.main()
