"""CPU regressions for advantage balancing, Gate, Repair, and global weighted-token normalization."""

import sys
import types
import unittest
from types import SimpleNamespace
from unittest import mock

import torch

import TRIAGE as triage


class FakeSample:
    def __init__(self, *, index, group_index, reward, response_length, loss_mask=None):
        self.index = index
        self.group_index = group_index
        self.reward = reward
        self.response_length = response_length
        self.loss_mask = loss_mask
        self.remove_sample = False

    def get_reward_value(self, _args):
        return self.reward


class TriageLossTest(unittest.TestCase):
    def setUp(self):
        self.old_refs = triage._GLOBAL_LENGTH_REFS
        self.old_seg_enable = triage.GATE_ENABLE
        self.old_repair_enable = triage.REPAIR_ENABLE
        self.old_repair_grad_scope = triage.REPAIR_SCOPE
        self.old_repair_start_step = triage.REPAIR_START_STEP
        self.old_recenter_enable = triage.RECENTER_ENABLE
        self.old_recenter_metrics = triage._LAST_RECENTER_METRICS
        self.old_seg_apply_scope = triage.GATE_SCOPE

    def tearDown(self):
        triage._GLOBAL_LENGTH_REFS = self.old_refs
        triage.GATE_ENABLE = self.old_seg_enable
        triage.REPAIR_ENABLE = self.old_repair_enable
        triage.REPAIR_SCOPE = self.old_repair_grad_scope
        triage.REPAIR_START_STEP = self.old_repair_start_step
        triage.RECENTER_ENABLE = self.old_recenter_enable
        triage._LAST_RECENTER_METRICS = self.old_recenter_metrics
        triage.GATE_SCOPE = self.old_seg_apply_scope

    @staticmethod
    def _recenter_args(*, group_size=2, num_groups=2):
        return SimpleNamespace(
            advantage_estimator="grpo",
            rewards_normalization=True,
            grpo_std_normalization=True,
            n_samples_per_prompt=group_size,
            rollout_batch_size=num_groups,
            global_batch_size=group_size * num_groups,
            rollout_max_response_len=20480,
        )

    def test_recenter_hook_uses_explicit_groups_after_shuffle(self):
        triage.RECENTER_ENABLE = True
        args = self._recenter_args()
        # Deliberately interleave the two prompt groups.  Contiguous reshape would be wrong.
        samples = [
            FakeSample(index=3, group_index=11, reward=1.0, response_length=100),
            FakeSample(index=0, group_index=10, reward=0.0, response_length=400),
            FakeSample(index=2, group_index=11, reward=0.0, response_length=400),
            FakeSample(index=1, group_index=10, reward=1.0, response_length=100),
        ]
        with mock.patch.object(triage, "_emit_recenter_metrics"):
            raw, adjusted = triage.triage_group_recenter_reward_postprocess(args, samples)

        self.assertEqual(raw, [1.0, 0.0, 0.0, 1.0])
        for group_index in (10, 11):
            values = [value for value, sample in zip(adjusted, samples, strict=True) if sample.group_index == group_index]
            self.assertAlmostEqual(sum(values) / len(values), 0.0, places=7)
        self.assertEqual(triage._LAST_RECENTER_METRICS["group_size"], 2.0)
        self.assertLess(triage._LAST_RECENTER_METRICS["post_group_mean_maxabs"], 1e-7)

    def test_recenter_hook_fails_on_missing_group_identity(self):
        triage.RECENTER_ENABLE = True
        args = self._recenter_args()
        samples = [
            FakeSample(index=i, group_index=(None if i == 0 else i // 2), reward=float(i % 2), response_length=100)
            for i in range(4)
        ]
        with self.assertRaisesRegex(RuntimeError, "missing group_index"):
            triage.triage_group_recenter_reward_postprocess(args, samples)

    def test_recenter_hook_fails_on_incomplete_group(self):
        triage.RECENTER_ENABLE = True
        args = self._recenter_args()
        samples = [
            FakeSample(index=0, group_index=10, reward=0.0, response_length=100),
            FakeSample(index=1, group_index=11, reward=0.0, response_length=100),
            FakeSample(index=2, group_index=11, reward=1.0, response_length=100),
            FakeSample(index=3, group_index=11, reward=1.0, response_length=100),
        ]
        with self.assertRaisesRegex(RuntimeError, "incomplete prompt groups"):
            triage.triage_group_recenter_reward_postprocess(args, samples)

    def test_recenter_removes_bias_from_negative_only_damping(self):
        # One long negative sample in each 8-way GRPO group is above global P70.  Damping only
        # that negative value must not introduce a positive group-mean drift.
        raw_rewards = []
        lengths = []
        groups = []
        for group_index in range(2):
            raw_rewards.extend([0.0] * 4 + [1.0] * 4)
            lengths.extend([2000.0, 500.0, 500.0, 500.0] + [500.0] * 4)
            groups.extend([group_index] * 8)

        adjusted, metrics = triage._group_recenter_advantages(
            raw_rewards,
            lengths,
            groups,
            expected_group_size=8,
            std_normalization=True,
        )
        self.assertGreater(metrics["pre_group_mean_maxabs"], 1e-4)
        self.assertLess(metrics["post_group_mean_maxabs"], 1e-7)
        for group_index in range(2):
            values = [value for value, group in zip(adjusted, groups, strict=True) if group == group_index]
            self.assertAlmostEqual(sum(values) / len(values), 0.0, places=7)

    def test_postprocess_counts_partial_masks_and_rejects_empty_response(self):
        fake_core = types.ModuleType("megatron.core")
        fake_core.mpu = SimpleNamespace()
        fake_megatron = types.ModuleType("megatron")
        fake_megatron.core = fake_core
        rollout = {
            "response_lengths": [100, 200, 300, 400],
            "loss_masks": [
                torch.ones(100),
                torch.cat([torch.ones(1), torch.zeros(199)]),
                torch.ones(300),
                torch.ones(400),
            ],
        }
        triage.GATE_ENABLE = False
        for key in ("log_probs", "rollout_log_probs", "advantages"):
            rollout[key] = [torch.zeros(length) for length in rollout["response_lengths"]]
        args = SimpleNamespace(global_batch_size=4, rollout_max_response_len=20480)
        with mock.patch.dict(
            sys.modules, {"megatron": fake_megatron, "megatron.core": fake_core}
        ):
            triage.triage_rollout_postprocess(args, 7, rollout)

        refs = triage._GLOBAL_LENGTH_REFS
        self.assertEqual(refs["rollout_id"], 7)
        self.assertEqual(refs["global_n"], 4)
        self.assertEqual(refs["active_n"], 4)
        self.assertEqual(refs["global_valid_segments"], 9)
        self.assertEqual(refs["global_token_count"], 801)
        self.assertEqual(refs["global_token_weight_units"], 80100)
        self.assertEqual(rollout[triage.TOKEN_WEIGHT_KEY][1].sum().item(), 100)
        rollout["loss_masks"][1].zero_()
        with self.assertRaisesRegex(RuntimeError, "no valid loss tokens"):
            triage._materialize_token_weight_units(rollout)

    def test_postprocess_fails_when_rollout_is_not_one_complete_update(self):
        fake_core = types.ModuleType("megatron.core")
        fake_core.mpu = SimpleNamespace()
        fake_megatron = types.ModuleType("megatron")
        fake_megatron.core = fake_core
        args = SimpleNamespace(global_batch_size=3, rollout_max_response_len=20480)
        rollout = {"response_lengths": [10, 20], "loss_masks": [torch.ones(10), torch.ones(20)]}
        triage.GATE_ENABLE = False
        for key in ("log_probs", "rollout_log_probs", "advantages"):
            rollout[key] = [torch.zeros(length) for length in rollout["response_lengths"]]
        with mock.patch.dict(
            sys.modules, {"megatron": fake_megatron, "megatron.core": fake_core}
        ):
            with self.assertRaisesRegex(RuntimeError, "exactly one complete update"):
                triage.triage_rollout_postprocess(args, 0, rollout)

    def test_global_references_are_dp_partition_invariant(self):
        two_ranks = [
            {"lengths": [100.0, 200.0], "total_n": 2, "valid_segments": 3},
            {"lengths": [300.0, 400.0], "total_n": 2, "valid_segments": 7},
        ]
        four_ranks = [
            {"lengths": [100.0], "total_n": 1, "valid_segments": 1},
            {"lengths": [200.0], "total_n": 1, "valid_segments": 2},
            {"lengths": [300.0], "total_n": 1, "valid_segments": 3},
            {"lengths": [400.0], "total_n": 1, "valid_segments": 4},
        ]
        for rank in two_ranks + four_ranks:
            rank["token_count"] = int(sum(rank["lengths"]))
            rank["token_weight_units"] = 100 * rank["token_count"]
        refs_two = triage._aggregate_length_metadata(
            two_ranks, expected_global_batch=4, horizon=20480, rollout_id=9
        )
        refs_four = triage._aggregate_length_metadata(
            four_ranks, expected_global_batch=4, horizon=20480, rollout_id=9
        )
        for key in ("global_n", "active_n", "global_valid_segments", "global_token_count", "global_token_weight_units"):
            self.assertEqual(refs_two[key], refs_four[key])

    def test_missing_global_metadata_fails(self):
        triage._GLOBAL_LENGTH_REFS = None
        with self.assertRaisesRegex(RuntimeError, "local micro-batch metadata fallback"):
            triage._require_global_length_refs(20480)

    def test_pseudo_huber_keeps_nonzero_bounded_gradient_on_severe_segment(self):
        delta = torch.tensor(-0.20, requires_grad=True)
        penalty, force, gap = triage._one_sided_pseudo_huber(delta, target=-0.03, beta=0.022360679775)
        penalty.backward()
        self.assertGreater(gap.item(), 0)
        self.assertGreater(abs(delta.grad.item()), 0)
        self.assertLessEqual(abs(delta.grad.item()), 0.022360679775 + 1e-7)
        self.assertAlmostEqual(abs(delta.grad.item()), force.item(), places=7)

    def test_adv_repair_scopes_preserve_forward_and_select_segment_gradient(self):
        base_delta = torch.tensor([[-0.2, 0.1], [-0.2, 0.1]])
        advantages = torch.tensor([[1.0, 1.0], [-1.0, -1.0]])
        mask = torch.ones_like(base_delta, dtype=torch.bool)
        expected_grad_masks = {
            "whole": torch.tensor([[True, True], [True, True]]),
            "adv_positive": torch.tensor([[True, True], [False, False]]),
            "q2": torch.tensor([[True, False], [False, False]]),
        }
        expected_forward = base_delta.mean(dim=-1)

        for scope, expected_grad_mask in expected_grad_masks.items():
            with self.subTest(scope=scope):
                live_delta = base_delta.clone().requires_grad_(True)
                segment_delta, grad_mask, valid = triage._scoped_repair_segment_delta(
                    live_delta,
                    advantages,
                    mask,
                    mask.float(),
                    grad_scope=scope,
                )
                self.assertTrue(torch.allclose(segment_delta.detach(), expected_forward))
                self.assertTrue(torch.equal(grad_mask, expected_grad_mask))
                self.assertTrue(torch.equal(valid, torch.tensor([True, True])))

                penalty, _, _ = triage._one_sided_pseudo_huber(
                    segment_delta,
                    target=-0.03,
                    beta=0.022360679775,
                )
                penalty.sum().backward()
                self.assertTrue(torch.equal(live_delta.grad != 0, expected_grad_mask))

    def test_adv_repair_scope_rejects_unknown_scope_and_shape_mismatch(self):
        values = torch.ones((1, 2))
        mask = values.bool()
        with self.assertRaisesRegex(ValueError, "Repair grad_scope"):
            triage._scoped_repair_segment_delta(
                values,
                values,
                mask,
                values,
                grad_scope="adv_negative",
            )
        with self.assertRaisesRegex(RuntimeError, "matching delta/advantage/mask shapes"):
            triage._scoped_repair_segment_delta(
                values,
                torch.ones((1, 1)),
                mask,
                values,
                grad_scope="whole",
            )

    def test_repair_start_step_schedule_is_inclusive(self):
        self.assertEqual(
            triage._effective_repair_coef(
                rollout_id=299, enabled=True, start_step=300, coef=0.03
            ),
            0.0,
        )
        self.assertEqual(
            triage._effective_repair_coef(
                rollout_id=300, enabled=True, start_step=300, coef=0.03
            ),
            0.03,
        )
        self.assertEqual(
            triage._effective_repair_coef(
                rollout_id=400, enabled=False, start_step=300, coef=0.03
            ),
            0.0,
        )
        with self.assertRaisesRegex(ValueError, "start_step"):
            triage._effective_repair_coef(
                rollout_id=0, enabled=True, start_step=-1, coef=0.03
            )

    def test_adv_positive_custom_loss_reports_only_applied_scope(self):
        triage.GATE_ENABLE = False
        triage.REPAIR_ENABLE = True
        triage.REPAIR_SCOPE = "adv_positive"
        triage.REPAIR_START_STEP = 300
        triage._GLOBAL_LENGTH_REFS = {
            "rollout_id": 300,
            "global_n": 2,
            "global_token_count": 4,
            "global_token_weight_units": 400,
            "active_n": 2,
            "global_valid_segments": 2,
            "dp_world_size": 1,
            "horizon": 20480.0,
        }

        current = [
            torch.tensor([-0.2, 0.1], requires_grad=True),
            torch.tensor([-0.2, 0.1], requires_grad=True),
        ]
        fake_loss_module = types.ModuleType("slime.backends.megatron_utils.loss")
        fake_loss_module.get_log_probs_and_entropy = lambda *args, **kwargs: (
            None,
            {"log_probs": current, "entropy": [torch.zeros(2), torch.zeros(2)]},
        )
        fake_ppo_module = types.ModuleType("slime.utils.ppo_utils")
        fake_ppo_module.compute_policy_loss = lambda ppo_kl, _adv, _lo, _hi: (
            0.0 * ppo_kl,
            torch.zeros_like(ppo_kl),
        )
        fake_ppo_module.compute_approx_kl = lambda *args, **kwargs: torch.tensor(0.0)
        batch = {
            "response_lengths": [2, 2],
            "advantages": [torch.ones(2), -torch.ones(2)],
            "log_probs": [torch.zeros(2), torch.zeros(2)],
            "rollout_log_probs": [torch.zeros(2), torch.zeros(2)],
            "unconcat_tokens": [torch.zeros(3), torch.zeros(3)],
            "total_lengths": [3, 3],
            "loss_masks": [torch.ones(2), torch.ones(2)],
        }
        batch[triage.TOKEN_WEIGHT_KEY] = [
            torch.full((length,), 100, dtype=torch.int32) for length in batch["response_lengths"]
        ]
        args = SimpleNamespace(
            calculate_per_token_loss=True,
            rollout_max_response_len=20480,
            eps_clip=0.2,
            eps_clip_high=0.28,
            entropy_coef=0.0,
            use_kl_loss=False,
            kl_loss_coef=0.0,
            kl_loss_type="low_var_kl",
            use_tis=False,
            use_rollout_logprobs=False,
            get_mismatch_metrics=False,
            custom_tis_function_path=None,
        )
        modules = {
            "slime.backends.megatron_utils.loss": fake_loss_module,
            "slime.utils.ppo_utils": fake_ppo_module,
        }
        with mock.patch.dict(sys.modules, modules):
            loss, metrics, normalizer = triage.triage_custom_loss(
                args,
                batch,
                torch.zeros(1, requires_grad=True),
                lambda values: values.sum(),
            )

        loss.backward()
        penalty, _, _ = triage._one_sided_pseudo_huber(
            torch.tensor(-0.05),
            target=triage.REPAIR_TARGET,
            beta=triage.REPAIR_HUBER_BETA,
        )
        self.assertTrue(torch.all(current[0].grad != 0))
        self.assertTrue(torch.equal(current[1].grad, torch.zeros(2)))
        # Megatron divides accumulated metric numerators by global weight units.
        self.assertTrue(torch.allclose(metrics["triage_repair_loss"] / normalizer, penalty / 2.0))
        self.assertAlmostEqual(
            (metrics["triage_repair_active_segment_fraction"] / normalizer).item(),
            0.5,
        )
        self.assertAlmostEqual(
            (metrics["triage_repair_start_step"] / normalizer).item(), 300.0
        )
        self.assertAlmostEqual(
            (metrics["triage_repair_current_step"] / normalizer).item(), 300.0
        )
        self.assertEqual(
            (metrics["triage_repair_schedule_active"] / normalizer).item(), 1.0
        )
        self.assertAlmostEqual(
            (metrics["triage_repair_effective_coef"] / normalizer).item(), triage.REPAIR_COEF
        )

        triage._GLOBAL_LENGTH_REFS["rollout_id"] = 299
        current = [
            torch.tensor([-0.2, 0.1], requires_grad=True),
            torch.tensor([-0.2, 0.1], requires_grad=True),
        ]
        with mock.patch.dict(sys.modules, modules):
            loss_before, metrics_before, _ = triage.triage_custom_loss(
                args,
                batch,
                torch.zeros(1, requires_grad=True),
                lambda values: values.sum(),
            )

        loss_before.backward()
        self.assertTrue(torch.equal(current[0].grad, torch.zeros(2)))
        self.assertTrue(torch.equal(current[1].grad, torch.zeros(2)))
        self.assertGreater(metrics_before["triage_repair_loss"].item(), 0.0)
        self.assertEqual(metrics_before["triage_repair_contribution"].item(), 0.0)
        self.assertEqual(
            (metrics_before["triage_repair_schedule_active"] / normalizer).item(), 0.0
        )
        self.assertEqual(
            (metrics_before["triage_repair_effective_coef"] / normalizer).item(), 0.0
        )

    def test_stock_tis_matches_slime_vanilla_formula(self):
        pg_loss = torch.tensor([1.0, 2.0, 3.0])
        train_old = torch.log(torch.tensor([0.5, 2.0, 4.0]))
        rollout = torch.zeros(3)

        corrected, ratio, clipfrac, absolute = triage._apply_stock_tis(
            pg_loss,
            train_old,
            rollout,
            lower=0.0,
            upper=2.0,
        )

        self.assertTrue(torch.allclose(ratio, torch.tensor([0.5, 2.0, 4.0])))
        self.assertTrue(torch.allclose(corrected, torch.tensor([0.5, 4.0, 6.0])))
        self.assertTrue(torch.equal(clipfrac, torch.tensor([0.0, 0.0, 1.0])))
        self.assertTrue(torch.allclose(absolute, torch.tensor([0.5, 1.0, 3.0])))

    def test_stock_tis_rejects_invalid_bounds(self):
        values = torch.ones(1)
        with self.assertRaisesRegex(ValueError, "invalid stock TIS bounds"):
            triage._apply_stock_tis(values, values, values, lower=2.0, upper=1.0)

    @staticmethod
    def _segment_weights(deltas, *, scope="q3", advantages=None, mask=None):
        deltas = torch.tensor([deltas], dtype=torch.float32)
        if advantages is None:
            advantages = [-1.0] * deltas.shape[-1]
        advantages = torch.tensor([advantages], dtype=torch.float32)
        if mask is None:
            mask = [True] * deltas.shape[-1]
        mask = torch.tensor([mask], dtype=torch.bool)
        return triage._gate_weights(
            deltas,
            advantages,
            mask,
            mask.to(torch.float32),
            apply_scope=scope,
        )

    def test_q3_scope_only_downweights_negative_delta_tokens(self):
        q3 = self._segment_weights([-1.2, -1.2, 0.1, 0.1], scope="q3")
        whole = self._segment_weights([-1.2, -1.2, 0.1, 0.1], scope="whole")
        self.assertTrue(torch.allclose(q3, torch.tensor([[0.3, 0.3, 1.0, 1.0]])))
        self.assertTrue(torch.allclose(whole, torch.full((1, 4), 0.3)))

    def test_q3_scope_keeps_segment_trigger_and_severe_precedence(self):
        weights = self._segment_weights([-4.0, -4.0, 0.0, 0.0], scope="q3")
        self.assertTrue(torch.allclose(weights, torch.tensor([[0.1, 0.1, 1.0, 1.0]])))

    def test_q3_scope_requires_negative_segment_advantage(self):
        weights = self._segment_weights(
            [-1.2, -1.2, 0.1, 0.1],
            scope="q3",
            advantages=[1.0, 1.0, 1.0, 1.0],
        )
        self.assertTrue(torch.equal(weights, torch.ones(1, 4)))

    def test_q3_scope_uses_strict_negative_token_boundary_and_mask(self):
        weights = self._segment_weights(
            [-2.0, 0.0, -100.0, -100.0],
            scope="q3",
            mask=[True, True, False, False],
        )
        self.assertTrue(torch.allclose(weights, torch.tensor([[0.3, 1.0, 1.0, 1.0]])))

    def test_segment_scope_rejects_unknown_value(self):
        with self.assertRaisesRegex(ValueError, "apply_scope"):
            self._segment_weights([-1.0], scope="negative-ish")

    def test_tail_ramp_maps_one_deep_token_to_intermediate_weight(self):
        weights = self._segment_weights([-6.1] + [0.1] * 63)

        self.assertAlmostEqual(weights[0, 0].item(), 0.55, places=6)
        self.assertTrue(torch.equal(weights[0, 1:], torch.ones_like(weights[0, 1:])))

    def test_tail_ramp_saturates_at_two_deep_tokens(self):
        weights = self._segment_weights([-6.1, -6.1] + [0.2] * 62)

        self.assertTrue(torch.allclose(weights[0, :2], torch.full((2,), triage.GATE_SEV_W)))
        self.assertTrue(torch.equal(weights[0, 2:], torch.ones_like(weights[0, 2:])))

    def test_tail_ramp_does_not_gate_nonnegative_advantage_segment(self):
        weights = self._segment_weights(
            [-7.0, -7.0] + [0.1] * 62,
            advantages=[0.0] * 64,
        )

        self.assertTrue(torch.equal(weights, torch.ones_like(weights)))

    def test_tail_ramp_rejects_shape_mismatch(self):
        values = torch.ones((1, 2))
        mask = values.bool()
        with self.assertRaisesRegex(RuntimeError, "matching delta/advantage/mask shapes"):
            triage._gate_weights(
                values,
                torch.ones((1, 1)),
                mask,
                values,
                apply_scope="q3",
            )

    def test_token_level_weight_units_are_materialized_from_complete_update(self):
        triage.GATE_ENABLE = True
        triage.GATE_SCOPE = "whole"
        rollout_data = {
            "response_lengths": [2, 4],
            "loss_masks": [torch.ones(2), torch.ones(4)],
            "log_probs": [torch.full((2,), -0.6), torch.zeros(4)],
            "rollout_log_probs": [torch.zeros(2), torch.zeros(4)],
            "advantages": [torch.full((2,), -1.0), torch.full((4,), -1.0)],
        }

        units, token_count, weight_units = triage._materialize_token_weight_units(rollout_data)

        self.assertEqual(token_count, 6)
        self.assertEqual(weight_units, 460)
        self.assertTrue(torch.equal(units[0], torch.tensor([30, 30], dtype=torch.int32)))
        self.assertTrue(
            torch.equal(units[1], torch.tensor([100, 100, 100, 100], dtype=torch.int32))
        )

    def test_token_level_weight_units_quantize_tail_ramp_k1(self):
        triage.GATE_ENABLE = True
        triage.GATE_SCOPE = "q3"
        rollout_data = {
            "response_lengths": [64],
            "loss_masks": [torch.ones(64)],
            "log_probs": [torch.cat([torch.full((1,), -6.1), torch.full((63,), 0.1)])],
            "rollout_log_probs": [torch.zeros(64)],
            "advantages": [torch.full((64,), -1.0)],
        }

        units, token_count, weight_units = triage._materialize_token_weight_units(rollout_data)

        self.assertEqual(token_count, 64)
        self.assertEqual(units[0][0].item(), 55)
        self.assertTrue(
            torch.equal(units[0][1:], torch.full((63,), 100, dtype=torch.int32))
        )
        self.assertEqual(weight_units, 55 + 63 * 100)

    def test_token_level_custom_loss_uses_one_global_mw_denominator(self):
        triage.GATE_ENABLE = True
        triage.REPAIR_ENABLE = False
        triage.GATE_SCOPE = "whole"
        triage._GLOBAL_LENGTH_REFS = {
            "rollout_id": 0,
            "global_n": 2,
            "active_n": 2,
            "global_valid_segments": 2,
            "global_token_count": 6,
            "global_token_weight_units": 460,
            "dp_world_size": 1,
            "horizon": 20480.0,
        }

        current = [torch.zeros(2, requires_grad=True), torch.zeros(4, requires_grad=True)]
        token_policy_loss = torch.tensor([[1.0, 3.0, 0.0, 0.0], [2.0, 4.0, 6.0, 8.0]])
        fake_loss_module = types.ModuleType("slime.backends.megatron_utils.loss")
        fake_loss_module.get_log_probs_and_entropy = lambda *args, **kwargs: (
            None,
            {"log_probs": current, "entropy": [torch.zeros(2), torch.zeros(4)]},
        )
        fake_ppo_module = types.ModuleType("slime.utils.ppo_utils")
        fake_ppo_module.compute_policy_loss = lambda _kl, _adv, _lo, _hi: (
            token_policy_loss,
            torch.zeros_like(token_policy_loss),
        )
        fake_ppo_module.compute_approx_kl = lambda *args, **kwargs: torch.tensor(0.0)
        batch = {
            "response_lengths": [2, 4],
            "advantages": [torch.full((2,), -1.0), torch.full((4,), -1.0)],
            "log_probs": [torch.zeros(2), torch.zeros(4)],
            "rollout_log_probs": [torch.zeros(2), torch.zeros(4)],
            "unconcat_tokens": [torch.zeros(3), torch.zeros(5)],
            "total_lengths": [3, 5],
            "loss_masks": [torch.ones(2), torch.ones(4)],
            triage.TOKEN_WEIGHT_KEY: [
                torch.tensor([30, 30], dtype=torch.int32),
                torch.tensor([100, 100, 100, 100], dtype=torch.int32),
            ],
        }
        args = SimpleNamespace(
            rollout_max_response_len=20480,
            calculate_per_token_loss=True,
            eps_clip=0.2,
            eps_clip_high=0.28,
            entropy_coef=0.0,
            use_kl_loss=False,
            kl_loss_coef=0.0,
            kl_loss_type="low_var_kl",
            use_tis=False,
            use_rollout_logprobs=False,
            get_mismatch_metrics=False,
        )
        modules = {
            "slime.backends.megatron_utils.loss": fake_loss_module,
            "slime.utils.ppo_utils": fake_ppo_module,
        }
        with mock.patch.dict(sys.modules, modules):
            numerator, metrics, normalizer = triage.triage_custom_loss(
                args, batch, torch.zeros(1, requires_grad=True), lambda values: values.sum()
            )

        expected = (
            30 * 1.0 + 30 * 3.0 + 100 * 2.0 + 100 * 4.0 + 100 * 6.0 + 100 * 8.0
        ) / 460
        self.assertEqual(normalizer.dtype, torch.int32)
        self.assertEqual(normalizer.item(), 460)
        self.assertAlmostEqual((numerator / normalizer).item(), expected, places=6)
        self.assertAlmostEqual((metrics["pg_loss"] / normalizer).item(), expected, places=6)

    def test_custom_loss_applies_stock_tis_and_reports_stock_metrics(self):
        triage.GATE_ENABLE = False
        triage.REPAIR_ENABLE = False
        triage._GLOBAL_LENGTH_REFS = {
            "rollout_id": 0,
            "global_n": 2,
            "global_token_count": 4,
            "global_token_weight_units": 400,
            "active_n": 2,
            "global_valid_segments": 2,
            "dp_world_size": 1,
            "horizon": 20480.0,
        }

        current = [torch.zeros(1, requires_grad=True), torch.zeros(3, requires_grad=True)]
        fake_loss_module = types.ModuleType("slime.backends.megatron_utils.loss")
        fake_loss_module.get_log_probs_and_entropy = lambda *args, **kwargs: (
            None,
            {"log_probs": current, "entropy": [torch.zeros(1), torch.zeros(3)]},
        )
        fake_ppo_module = types.ModuleType("slime.utils.ppo_utils")
        fake_ppo_module.compute_policy_loss = lambda _kl, adv, _lo, _hi: (
            adv,
            torch.zeros_like(adv),
        )
        fake_ppo_module.compute_approx_kl = lambda *args, **kwargs: torch.tensor(0.0)

        train_ratios = [torch.tensor([0.5]), torch.tensor([1.0, 2.0, 4.0])]
        batch = {
            "response_lengths": [1, 3],
            "advantages": [torch.ones(1), torch.ones(3)],
            "log_probs": [ratio.log() for ratio in train_ratios],
            "rollout_log_probs": [torch.zeros(1), torch.zeros(3)],
            "unconcat_tokens": [torch.zeros(2), torch.zeros(4)],
            "total_lengths": [2, 4],
            "loss_masks": [torch.ones(1), torch.ones(3)],
        }
        batch[triage.TOKEN_WEIGHT_KEY] = [
            torch.full((length,), 100, dtype=torch.int32) for length in batch["response_lengths"]
        ]
        args = SimpleNamespace(
            calculate_per_token_loss=True,
            rollout_max_response_len=20480,
            eps_clip=0.2,
            eps_clip_high=0.28,
            entropy_coef=0.0,
            use_kl_loss=False,
            kl_loss_coef=0.0,
            kl_loss_type="low_var_kl",
            use_tis=True,
            tis_clip_low=0.0,
            tis_clip=2.0,
            use_rollout_logprobs=False,
            get_mismatch_metrics=False,
            custom_tis_function_path=None,
        )

        modules = {
            "slime.backends.megatron_utils.loss": fake_loss_module,
            "slime.utils.ppo_utils": fake_ppo_module,
        }
        with mock.patch.dict(sys.modules, modules):
            loss, metrics, normalizer = triage.triage_custom_loss(
                args, batch, torch.zeros(1, requires_grad=True), lambda values: values.sum()
            )

        self.assertEqual(normalizer.item(), 400)
        self.assertAlmostEqual((loss / normalizer).item(), 5.5 / 4)
        self.assertTrue(torch.allclose(metrics["pg_loss"], loss))
        for key, expected in {"tis": 7.5 / 4, "tis_clipfrac": 1 / 4,
                              "tis_abs": 4.5 / 4, "ois": 3.75 / 4,
                              "triage_tis_enabled": 1}.items():
            self.assertAlmostEqual((metrics[key] / normalizer).item(), expected)


if __name__ == "__main__":
    unittest.main()
