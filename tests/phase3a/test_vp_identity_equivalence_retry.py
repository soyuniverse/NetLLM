import json
import unittest
from pathlib import Path


RESULT_PATH = Path(
    "/workspace/NetLLM/experiments/vp/phase3a_retry_runtime/identity_equivalence.json"
)


class VPIdentityEquivalenceRetryResultTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not RESULT_PATH.is_file():
            raise unittest.SkipTest(f"Phase 3A retry result is missing: {RESULT_PATH}")
        cls.result = json.loads(RESULT_PATH.read_text())

    def test_original_disabled_identity_equivalence(self):
        self.assertTrue(self.result["success"])
        self.assertEqual(
            set(self.result["comparisons"]),
            {
                "original_vs_disabled",
                "original_vs_identity",
                "disabled_vs_identity",
            },
        )
        for comparison in self.result["comparisons"].values():
            self.assertLessEqual(comparison["max_absolute_difference"], 1e-7)
            self.assertTrue(comparison["within_tolerance"])
            self.assertEqual(comparison["rtol"], 0.0)
            self.assertEqual(comparison["atol"], 1e-7)
            self.assertEqual(len(comparison["max_difference_index"]), 3)
            self.assertIn("left_value_at_max", comparison)
            self.assertIn("right_value_at_max", comparison)

    def test_path_contracts_match(self):
        expected_lengths = list(range(10, 30))
        self.assertEqual(
            set(self.result["paths"]),
            {"original", "selectable_disabled", "selectable_identity"},
        )
        for path in self.result["paths"].values():
            output = path["output"]
            trace = path["trace"]
            self.assertEqual(output["shape"], [1, 20, 3])
            self.assertEqual(output["dtype"], "torch.float32")
            self.assertEqual(output["device"], "cuda:0")
            self.assertEqual(len(output["sha256"]), 64)
            self.assertTrue(output["finite"])
            self.assertEqual(trace["sequence_lengths"], expected_lengths)
            self.assertEqual(trace["plm_forward_count"], 20)
            self.assertFalse(any(trace["past_key_values_passed"]))
            self.assertFalse(trace["cache_reused"])

    def test_identity_metadata(self):
        selection = self.result["identity_selection_output"]
        identity_trace = self.result["extension"]["identity_trace"]
        disabled_trace = self.result["extension"]["disabled_trace"]
        self.assertEqual(selection["original_length"], 10)
        self.assertEqual(selection["selected_length"], 10)
        self.assertEqual(selection["selected_indices"], list(range(10)))
        self.assertIsNone(selection["scores"])
        self.assertTrue(selection["embeddings_same_object"])
        self.assertTrue(selection["attention_mask_same_object"])
        self.assertEqual(identity_trace["selector_call_count"], 1)
        self.assertFalse(identity_trace["selector_applied_to_feedback"])
        self.assertEqual(disabled_trace["selector_call_count"], 0)


if __name__ == "__main__":
    unittest.main()
