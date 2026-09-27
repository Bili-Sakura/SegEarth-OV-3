import unittest

from segearthov3.prompts import normalize_text_prompts
from segearthov3.vocabularies import LOVEDA, OPENEARTHMAP, STPLS3D


class PromptNormalizationTests(unittest.TestCase):
    def test_flat_list(self):
        spec = normalize_text_prompts(["building", "road", "water"])
        self.assertEqual(spec.query_words, ["building", "road", "water"])
        self.assertEqual(spec.query_idx, [0, 1, 2])
        self.assertEqual(spec.num_classes, 3)
        self.assertEqual(spec.num_queries, 3)

    def test_comma_synonyms(self):
        spec = normalize_text_prompts(["building,house", "road"])
        self.assertEqual(spec.query_words, ["building", "house", "road"])
        self.assertEqual(spec.query_idx, [0, 0, 1])
        self.assertEqual(spec.class_prompts, ["building,house", "road"])

    def test_nested_synonym_groups(self):
        spec = normalize_text_prompts([["building", "house"], "road"])
        self.assertEqual(spec.query_words, ["building", "house", "road"])
        self.assertEqual(spec.query_idx, [0, 0, 1])

    def test_rejects_single_string(self):
        with self.assertRaises(TypeError):
            normalize_text_prompts("building")

    def test_rejects_empty_list(self):
        with self.assertRaises(ValueError):
            normalize_text_prompts([])

    def test_loveda_preset_expands_synonyms(self):
        spec = normalize_text_prompts(LOVEDA)
        self.assertEqual(spec.num_classes, 7)
        self.assertIn("house", spec.query_words)
        self.assertGreater(spec.num_queries, spec.num_classes)

    def test_openearthmap_and_stpls3d_presets(self):
        oem = normalize_text_prompts(OPENEARTHMAP)
        self.assertEqual(oem.num_classes, 9)
        stpls = normalize_text_prompts(STPLS3D)
        self.assertEqual(stpls.num_classes, 6)
        self.assertEqual(stpls.query_idx[0], stpls.query_idx[1])


if __name__ == "__main__":
    unittest.main()
