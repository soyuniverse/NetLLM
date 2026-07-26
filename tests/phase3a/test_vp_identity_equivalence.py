import json
import os
import unittest
from pathlib import Path


PROJECT_ROOT = Path(
    os.environ.get("NETLLM_PROJECT_ROOT", Path(__file__).resolve().parents[2])
).resolve()
RESULT_PATH = Path(
    os.environ.get(
        "NETLLM_PHASE3A_RESULT",
        PROJECT_ROOT
        / "experiments/vp/phase3a_final_runtime/identity_equivalence.json",
    )
).resolve()


class VPIdentityEquivalenceResultTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not RESULT_PATH.is_file():
            raise unittest.SkipTest(f"Phase 3A runtime result is missing: {RESULT_PATH}")
        cls.result = json.loads(RESULT_PATH.read_text())

    def test_original_disabled_identity_equivalence(self):
        self.assertTrue(self.result["success"])
        for comparison in self.result["comparisons"].values():
            self.assertLessEqual(comparison["max_absolute_difference"], 1e-7)
            self.assertTrue(comparison["within_tolerance"])

    def test_path_contracts_match(self):
        expected_lengths = list(range(10, 30))
        for path in self.result["paths"].values():
            self.assertEqual(path["output"]["shape"], [1, 20, 3])
            self.assertTrue(path["output"]["finite"])
            self.assertEqual(path["trace"]["sequence_lengths"], expected_lengths)
            self.assertEqual(path["trace"]["plm_forward_count"], 20)
            self.assertFalse(any(path["trace"]["past_key_values_passed"]))

    def test_identity_metadata(self):
        selection = self.result["identity_selection_output"]
        self.assertEqual(selection["original_length"], 10)
        self.assertEqual(selection["selected_length"], 10)
        self.assertEqual(selection["selected_indices"], list(range(10)))
        self.assertIsNone(selection["scores"])
        self.assertTrue(selection["embeddings_same_object"])
        self.assertTrue(selection["attention_mask_same_object"])


if __name__ == "__main__":
    unittest.main()
