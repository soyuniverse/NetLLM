import unittest

import torch

from netllm_litevlm.selectors import AdaptiveKSelector, IdentitySelector, RecentKSelector
from netllm_litevlm.selectors.adaptive_k import _DENORMALIZE_SCALE_DEG, history_motion_speed


def _history_with_constant_step(step_deg: float, steps: int = 9) -> torch.Tensor:
    """[1, steps+1, 3] RAW-DEGREE history where every channel moves by
    exactly step_deg degrees per step -> history_motion_speed(raw) ==
    step_deg. For testing `history_motion_speed` directly."""
    values = torch.arange(steps + 1, dtype=torch.float32) * step_deg
    return values.view(1, steps + 1, 1).repeat(1, 1, 3)


def _normalized_history_with_constant_raw_step(step_deg: float, steps: int = 9) -> torch.Tensor:
    """Same raw-degree step pattern as `_history_with_constant_step`, but
    pre-divided by `_DENORMALIZE_SCALE_DEG` so that feeding this through
    AdaptiveKSelector (which denormalizes internally, matching what the
    real pipelines' `history` argument actually is) reproduces an
    effective raw-degree step of step_deg."""
    raw = _history_with_constant_step(step_deg, steps)
    return raw / _DENORMALIZE_SCALE_DEG


class HistoryMotionSpeedTest(unittest.TestCase):
    def test_constant_step_matches_step_size(self):
        history = _history_with_constant_step(2.5)
        self.assertAlmostEqual(history_motion_speed(history), 2.5, places=5)

    def test_zero_motion(self):
        history = torch.zeros(1, 10, 3)
        self.assertEqual(history_motion_speed(history), 0.0)

    def test_wraparound_treated_as_small(self):
        # 179 -> -179 is an 18-degree change in wrapped space, not 358.
        history = torch.tensor([[[179.0, 0.0, 0.0], [-179.0, 0.0, 0.0]]])
        self.assertAlmostEqual(history_motion_speed(history), 2.0 / 3.0, places=5)


class AdaptiveKSelectorGateTest(unittest.TestCase):
    """CPU gate: low-speed input -> matches RecentK-2 exactly; high-speed
    input -> matches Identity exactly; mixed input -> switches at the
    exact threshold boundary."""

    def setUp(self):
        self.v_low = 2.0
        self.v_high = 5.0
        self.selector = AdaptiveKSelector(v_low=self.v_low, v_high=self.v_high, k_low=2, k_mid=4, k_high=10)
        self.embeddings = torch.arange(10 * 4, dtype=torch.float32).view(1, 10, 4)
        self.mask = torch.ones(1, 10, dtype=torch.long)

    def _run(self, history):
        return self.selector(
            self.embeddings, self.mask, context={"task": "t", "history": history}
        )

    def test_low_speed_matches_recent_k2(self):
        history = _normalized_history_with_constant_raw_step(0.5)  # well below v_low
        output = self._run(history)
        reference = RecentKSelector(2)(self.embeddings, self.mask, {})

        self.assertTrue(torch.equal(output.embeddings, reference.embeddings))
        self.assertTrue(torch.equal(output.attention_mask, reference.attention_mask))
        self.assertEqual(output.selected_indices.tolist(), reference.selected_indices.tolist())
        self.assertEqual(output.selected_length, 2)
        self.assertEqual(self.selector.last_k, 2)

    def test_high_speed_matches_identity(self):
        history = _normalized_history_with_constant_raw_step(50.0)  # well above v_high
        output = self._run(history)
        reference = IdentitySelector()(self.embeddings, self.mask, {})

        self.assertTrue(torch.equal(output.embeddings, reference.embeddings))
        self.assertTrue(torch.equal(output.attention_mask, reference.attention_mask))
        self.assertEqual(output.selected_indices.tolist(), reference.selected_indices.tolist())
        self.assertEqual(output.selected_length, 10)
        self.assertEqual(self.selector.last_k, 10)

    def test_mixed_speed_switches_exactly_at_threshold(self):
        # Exactly at v_low -> still the low bucket (k_low), condition is <=.
        at_v_low = _normalized_history_with_constant_raw_step(self.v_low)
        self._run(at_v_low)
        self.assertEqual(self.selector.last_k, 2)

        # Just above v_low -> mid bucket.
        just_above_v_low = _normalized_history_with_constant_raw_step(self.v_low + 1e-3)
        self._run(just_above_v_low)
        self.assertEqual(self.selector.last_k, 4)

        # Exactly at v_high -> still the mid bucket.
        at_v_high = _normalized_history_with_constant_raw_step(self.v_high)
        self._run(at_v_high)
        self.assertEqual(self.selector.last_k, 4)

        # Just above v_high -> high bucket.
        just_above_v_high = _normalized_history_with_constant_raw_step(self.v_high + 1e-3)
        self._run(just_above_v_high)
        self.assertEqual(self.selector.last_k, 10)

    def test_requires_history_in_context(self):
        with self.assertRaises(ValueError):
            self.selector(self.embeddings, self.mask, context={"task": "t"})
        with self.assertRaises(ValueError):
            self.selector(self.embeddings, self.mask, context=None)

    def test_rejects_invalid_thresholds_and_k(self):
        with self.assertRaises(ValueError):
            AdaptiveKSelector(v_low=5.0, v_high=2.0)
        with self.assertRaises(ValueError):
            AdaptiveKSelector(v_low=1.0, v_high=2.0, k_low=4, k_mid=2, k_high=10)
        with self.assertRaises(ValueError):
            AdaptiveKSelector(v_low=1.0, v_high=2.0, k_low=0)


if __name__ == "__main__":
    unittest.main()
