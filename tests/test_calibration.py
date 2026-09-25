"""Frozen calibration, strict masks, and default-off runtime integration."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import torch
import TRIAGE as triage
import triage_calibration as calibration


class CalibrationTest(unittest.TestCase):
    def records(self):
        return [{"rollout_id": i // 50, "response_id": str(i), "delta": [-0.01 * (i + 1)] * 4,
                 "advantage": -1.0, "loss_mask": [1, 1, 1, 0]} for i in range(100)]

    def state(self):
        return calibration.calibrate(self.records(), window_size=4, rounds=2, target_fpr=0.1, quantile_full=0.99)

    def test_calibration_has_bounded_reference_rate_and_is_order_invariant(self):
        state = self.state()
        other = calibration.calibrate(list(reversed(self.records())), window_size=4, rounds=2,
                                      target_fpr=0.1, quantile_full=0.99)
        self.assertEqual(state, other)
        self.assertLessEqual(state["reference"]["empirical_union_rate"], 0.1)
        self.assertEqual(state["reference"]["eligible_segments"], 100)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "state.json"
            path.write_text(json.dumps(state))
            self.assertEqual(calibration.load_state(path, window_size=4), state)
            legacy = copy.deepcopy(state)
            legacy["algorithm"] = "triage_m2_reference_v1"
            path.write_text(json.dumps(legacy))
            self.assertEqual(calibration.load_state(path, window_size=4), state)
            with self.assertRaisesRegex(ValueError, "segment size"):
                calibration.load_state(path, window_size=64)

    def test_invalid_reference_and_state_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "every rollout"):
            calibration.calibrate(self.records()[:50], window_size=4, rounds=2)
        with self.assertRaisesRegex(ValueError, "duplicate"):
            calibration.calibrate(self.records() + self.records()[:1], window_size=4, rounds=2)
        records = self.records()
        records[0]["delta"][0] = float("nan")
        with self.assertRaisesRegex(ValueError, "finite"):
            calibration.calibrate(records, window_size=4, rounds=2)
        state = self.state()
        state["tau_deep"] = float("inf")
        with self.assertRaises(ValueError):
            calibration.validate_state(state, window_size=4)

    def test_calibrated_gate_preserves_scope_mask_and_positive_advantage(self):
        state = self.state()
        delta = torch.tensor([[-4.0, 0.1, -4.0, -100.0], [-4.0, -4.0, 0.1, 0.1]])
        advantage = torch.tensor([[-1.0] * 4, [1.0] * 4])
        mask = torch.tensor([[True, True, True, False], [True] * 4])
        q3 = calibration.segment_weights(delta, advantage, mask, mask.float(), state=state, apply_scope="q3")
        whole = calibration.segment_weights(delta, advantage, mask, mask.float(), state=state, apply_scope="whole")
        self.assertAlmostEqual(q3[0, 0].item(), 0.1)
        self.assertEqual(q3[0, 1].item(), 1.0)
        self.assertLess(whole[0, 1].item(), 1.0)
        self.assertEqual(q3[0, 3].item(), 1.0)
        self.assertTrue(torch.equal(q3[1], torch.ones(4)))

    def test_runtime_materializes_calibrated_weights_as_detached_units(self):
        state = self.state()
        rollout = {"response_lengths": [4], "loss_masks": [torch.ones(4)],
                   "log_probs": [torch.tensor([-4.0, 0.1, -4.0, 0.1], requires_grad=True)],
                   "rollout_log_probs": [torch.zeros(4)], "advantages": [-torch.ones(4)]}
        frozen = copy.deepcopy(state)
        with mock.patch.multiple(triage, CALIBRATION_STATE=state, GATE_ENABLE=True, GATE_SCOPE="q3"):
            units, tokens, normalizer = triage._materialize_token_weight_units(rollout)
        self.assertEqual(units[0].tolist(), [10, 100, 10, 100])
        self.assertFalse(units[0].requires_grad)
        self.assertEqual((tokens, normalizer), (4, 220))
        self.assertEqual(state, frozen)

    def test_reference_export_never_overwrites_and_contains_no_prompt_text(self):
        rollout = {"response_lengths": [2], "log_probs": [torch.tensor([-0.1, -0.2])],
                   "rollout_log_probs": [torch.zeros(2)], "advantages": [-torch.ones(2)]}
        with tempfile.TemporaryDirectory() as directory:
            calibration.export_reference(directory, 0, 3, rollout)
            record = json.loads(next(Path(directory).glob('*.jsonl')).read_text())
            self.assertEqual(record["response_id"], "3:0")
            self.assertEqual(record["loss_mask"], [1.0, 1.0])
            with self.assertRaises(FileExistsError):
                calibration.export_reference(directory, 0, 3, rollout)


if __name__ == "__main__":
    unittest.main()
