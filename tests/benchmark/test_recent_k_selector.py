import unittest

import torch

from netllm_litevlm.selectors import RecentKSelector


class RecentKSelectorTest(unittest.TestCase):
    def test_keeps_recent_tokens_in_order(self):
        embeddings = torch.arange(40, dtype=torch.float32).view(1, 10, 4)
        mask = torch.arange(10, dtype=torch.long).view(1, 10)
        output = RecentKSelector(4)(embeddings, mask, {"sample": 0})

        self.assertTrue(torch.equal(output.embeddings, embeddings[:, 6:, :]))
        self.assertTrue(torch.equal(output.attention_mask, mask[:, 6:]))
        self.assertEqual(output.selected_indices.tolist(), [6, 7, 8, 9])
        self.assertEqual(output.original_length, 10)
        self.assertEqual(output.selected_length, 4)
        self.assertEqual(output.metadata["selection_policy"], "most_recent")
        self.assertTrue(output.metadata["preserves_order"])

    def test_supports_none_mask(self):
        embeddings = torch.zeros(2, 8, 3)
        output = RecentKSelector(2)(embeddings)
        self.assertEqual(output.embeddings.shape, (2, 2, 3))
        self.assertIsNone(output.attention_mask)
        self.assertEqual(output.selected_indices.tolist(), [6, 7])

    def test_rejects_invalid_k(self):
        with self.assertRaises(ValueError):
            RecentKSelector(0)
        with self.assertRaises(ValueError):
            RecentKSelector(True)
        with self.assertRaises(ValueError):
            RecentKSelector(11)(torch.zeros(1, 10, 3))


if __name__ == "__main__":
    unittest.main()
