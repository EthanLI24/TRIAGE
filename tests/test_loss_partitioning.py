"""Exercise the actual loss across uneven micro-batches, masks, and Repair scopes."""

import sys
import types
import unittest
from types import SimpleNamespace
from unittest import mock

import torch

import TRIAGE as triage


def evaluate(module, *, scope, segment_scope, partition, step=3, repair_enabled=True):
    """Run one synthetic update through the materializer and custom loss."""
    generator = torch.Generator().manual_seed(17)
    lengths = [5, 129, 258]
    current = [(-0.2 + 0.25 * torch.randn(n, generator=generator)).requires_grad_()
               for n in lengths]
    # Exercise both tail-ramp weights and the final partial Repair segment.
    with torch.no_grad():
        current[1][0] = -6.1
        current[2][:2] = -6.1
    masks = [torch.ones(n) for n in lengths]
    masks[0][-1] = 0
    masks[2][64:128] = 0
    advantages = [torch.ones(5), -torch.ones(129), torch.ones(258)]
    advantages[2][:128] = -1
    advantages[2][256:] = 0
    rollout = {
        "response_lengths": lengths,
        "loss_masks": masks,
        "log_probs": [v.detach() for v in current],
        "rollout_log_probs": [torch.zeros(n) for n in lengths],
        "advantages": advantages,
    }
    args = SimpleNamespace(
        rollout_max_response_len=512, calculate_per_token_loss=True,
        eps_clip=0.2, eps_clip_high=0.28, entropy_coef=0.0,
        use_kl_loss=False, kl_loss_coef=0.0, use_tis=True,
        tis_clip_low=0.0, tis_clip=2.0, use_rollout_logprobs=False,
    )
    loss_module = types.ModuleType("slime.backends.megatron_utils.loss")
    ppo_module = types.ModuleType("slime.utils.ppo_utils")

    def policy_loss(kl, adv, lo, hi):
        ratio = (-kl).exp()
        unclipped = -adv * ratio
        clipped = -adv * ratio.clamp(1 - lo, 1 + hi)
        return torch.maximum(unclipped, clipped), (clipped > unclipped).float()

    ppo_module.compute_policy_loss = policy_loss
    ppo_module.compute_approx_kl = lambda *a, **kw: torch.tensor(0.0)
    modules = {"slime.backends.megatron_utils.loss": loss_module,
               "slime.utils.ppo_utils": ppo_module}
    controls = dict(GATE_ENABLE=True, GATE_SCOPE=segment_scope,
                    REPAIR_ENABLE=repair_enabled, REPAIR_SCOPE=scope,
                    REPAIR_START_STEP=3, _GLOBAL_LENGTH_REFS=None)
    with mock.patch.multiple(module, **controls), mock.patch.dict(sys.modules, modules):
        units, token_count, weight_units = module._materialize_token_weight_units(rollout)
        active_lengths, total_n, segments = module._rollout_length_metadata(rollout)
        module._GLOBAL_LENGTH_REFS = module._aggregate_length_metadata(
            [dict(lengths=active_lengths, total_n=total_n, valid_segments=segments,
                  token_count=token_count, token_weight_units=weight_units)],
            expected_global_batch=3, horizon=512, rollout_id=step,
        )
        total_loss = torch.zeros(())
        total_normalizer = 0
        metrics = {}
        offset = 0
        for count in partition:
            stop = offset + count
            batch = {key: value[offset:stop] for key, value in rollout.items()}
            batch[module.TOKEN_WEIGHT_KEY] = units[offset:stop]
            batch["total_lengths"] = [n + 1 for n in lengths[offset:stop]]
            batch["unconcat_tokens"] = [torch.zeros(n) for n in batch["total_lengths"]]
            local = current[offset:stop]
            loss_module.get_log_probs_and_entropy = lambda *a, **kw: (
                None, {"log_probs": local, "entropy": [torch.zeros_like(v) for v in local]}
            )
            mask = torch.cat(masks[offset:stop])
            numerator, reported, normalizer = module.triage_custom_loss(
                args, batch, torch.zeros(1, requires_grad=True),
                lambda values: (values * mask).sum(),
            )
            total_loss = total_loss + numerator
            total_normalizer += int(normalizer)
            for key, value in reported.items():
                metrics[key] = metrics.get(key, 0) + value
            offset = stop
        assert offset == len(lengths)
        assert total_normalizer == weight_units
        loss = total_loss / total_normalizer
        loss.backward()
        return (loss.detach(), torch.cat([v.grad for v in current]),
                {key: value / total_normalizer for key, value in metrics.items()})


class LossPartitionTest(unittest.TestCase):
    def test_loss_gradients_and_metrics_are_partition_invariant(self):
        for scope in ("whole", "adv_positive", "q2"):
            for segment_scope in ("whole", "q3"):
                for step, enabled in ((2, True), (3, True), (3, False)):
                    baseline = evaluate(triage, scope=scope, segment_scope=segment_scope,
                                        partition=[3], step=step, repair_enabled=enabled)
                    for partition in ([1, 2], [1, 1, 1]):
                        with self.subTest(scope=scope, gate=segment_scope, step=step,
                                          enabled=enabled, partition=partition):
                            actual = evaluate(triage, scope=scope, segment_scope=segment_scope,
                                              partition=partition, step=step,
                                              repair_enabled=enabled)
                            torch.testing.assert_close(actual[0], baseline[0])
                            torch.testing.assert_close(actual[1], baseline[1])
                            for key in baseline[2]:
                                torch.testing.assert_close(actual[2][key], baseline[2][key],
                                                           msg=lambda msg: f"{key}: {msg}")


if __name__ == "__main__":
    unittest.main()
