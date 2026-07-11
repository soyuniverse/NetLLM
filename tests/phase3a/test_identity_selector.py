import sys
import unittest
from pathlib import Path

import torch


PROJECT_ROOT = Path("/workspace/NetLLM")
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from netllm_litevlm.selectors import IdentitySelector


class IdentitySelectorTest(unittest.TestCase):
    def devices(self):
        devices = [torch.device("cpu")]
        if torch.cuda.is_available():
            devices.append(torch.device("cuda:0"))
        return devices

    def test_preserves_tensor_contract_without_in_place_mutation(self):
        selector = IdentitySelector()
        for device in self.devices():
            with self.subTest(device=str(device)):
                embeddings = torch.randn(
                    1, 10, 1024, device=device, dtype=torch.float32, requires_grad=True
                )
                mask = torch.ones(1, 10, device=device, dtype=torch.long)
                embeddings_before = embeddings.detach().clone()
                mask_before = mask.clone()

                output = selector(embeddings, mask, context={"sample": "unit"})

                self.assertIs(output.embeddings, embeddings)
                self.assertIs(output.attention_mask, mask)
                self.assertTrue(torch.equal(embeddings, embeddings_before))
                self.assertTrue(torch.equal(mask, mask_before))
                self.assertEqual(output.embeddings.shape, (1, 10, 1024))
                self.assertEqual(output.embeddings.dtype, embeddings.dtype)
                self.assertEqual(output.embeddings.device, embeddings.device)
                self.assertEqual(output.embeddings.requires_grad, embeddings.requires_grad)
                self.assertEqual(output.selected_indices.tolist(), list(range(10)))
                self.assertEqual(output.selected_indices.dtype, torch.long)
                self.assertEqual(output.selected_indices.device, embeddings.device)
                self.assertEqual(output.original_length, 10)
                self.assertEqual(output.selected_length, 10)
                self.assertIsNone(output.scores)
                self.assertTrue(output.metadata["preserves_order"])
                self.assertEqual(output.metadata["context"], {"sample": "unit"})

    def test_preserves_none_attention_mask(self):
        embeddings = torch.randn(1, 4, 8)
        output = IdentitySelector()(embeddings, None)
        self.assertIs(output.embeddings, embeddings)
        self.assertIsNone(output.attention_mask)
        self.assertEqual(output.selected_indices.tolist(), [0, 1, 2, 3])

    def test_rejects_invalid_shapes(self):
        selector = IdentitySelector()
        with self.assertRaises(ValueError):
            selector(torch.randn(10, 1024), None)
        with self.assertRaises(ValueError):
            selector(torch.randn(1, 10, 1024), torch.ones(1, 9))


if __name__ == "__main__":
    unittest.main()
