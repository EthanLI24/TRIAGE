#!/usr/bin/env python3
"""Patch the pinned Slime checkout for custom per-token normalizers.

The patch is intentionally tiny and hash-gated.  It lets a custom loss return
``(loss_numerator, metrics, local_normalizer)`` and forwards one extra per-sample
field through the training DataIterator.  Stock losses and sequence-level custom
losses keep their original two-value contract and behavior.
"""

from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path


DEFAULT_SLIME_ROOT = Path(
    os.environ.get("SLIME_ROOT", Path(__file__).resolve().parents[2] / "external" / "slime")
)


STOCK_SHA256 = {
    "slime/backends/megatron_utils/loss.py": "d0c309032940315afa57d657ba5e16c0e004822cf722d1afce79606103dbbbe6",
    "slime/backends/megatron_utils/model.py": "c18e5450c564cdaa5b8c69756ad38e31d088ee89ed7d268742466deab7aa2296",
}
MARKER = "SLIME_TRIAGE_TOKENMEAN_CUSTOM_NORMALIZER_V1"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _replace_once(text: str, old: str, new: str, *, path: Path) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"expected one patch anchor in {path}, found {count}")
    return text.replace(old, new, 1)


def patch_loss(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    if MARKER in text:
        return
    old_call = '''    if args.recompute_loss_function:
        loss, log = checkpoint(func, args, batch, logits, sum_of_sample_mean)
    else:
        loss, log = func(args, batch, logits, sum_of_sample_mean)
'''
    new_call = f'''    # {MARKER}: custom losses may supply an exact local token normalizer.
    if args.recompute_loss_function:
        loss_result = checkpoint(func, args, batch, logits, sum_of_sample_mean)
    else:
        loss_result = func(args, batch, logits, sum_of_sample_mean)

    custom_loss_normalizer = None
    if len(loss_result) == 2:
        loss, log = loss_result
    elif len(loss_result) == 3:
        if args.loss_type != "custom_loss" or not args.calculate_per_token_loss:
            raise RuntimeError(
                "a custom loss normalizer is supported only for token-level custom_loss"
            )
        loss, log, custom_loss_normalizer = loss_result
        if (
            not isinstance(custom_loss_normalizer, torch.Tensor)
            or custom_loss_normalizer.numel() != 1
            or custom_loss_normalizer.requires_grad
            or custom_loss_normalizer.dtype != torch.int32
            or custom_loss_normalizer.item() <= 0
        ):
            raise RuntimeError(
                "custom token normalizer must be a detached positive scalar torch.int32 tensor"
            )
    else:
        raise RuntimeError(f"loss function returned {{len(loss_result)}} values; expected 2 or 3")

    loss_normalizer = (
        custom_loss_normalizer
        if custom_loss_normalizer is not None
        else (num_tokens if args.calculate_per_token_loss else torch.tensor(1, device=logits.device))
    )
'''
    text = _replace_once(text, old_call, new_call, path=path)
    old_return = '''        (num_tokens if args.calculate_per_token_loss else torch.tensor(1, device=logits.device)),
        {
            "keys": list(log.keys()),
            "values": torch.tensor(
                [
                    num_samples if not args.calculate_per_token_loss else num_tokens,
                ]
'''
    new_return = '''        loss_normalizer,
        {
            "keys": list(log.keys()),
            "values": torch.tensor(
                [
                    num_samples if not args.calculate_per_token_loss else loss_normalizer,
                ]
'''
    text = _replace_once(text, old_return, new_return, path=path)
    path.write_text(text, encoding="utf-8")


def patch_model(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    if MARKER in text:
        return
    old = '''                "rollout_log_probs",
                "max_seq_lens",
                "teacher_log_probs",
'''
    new = f'''                "rollout_log_probs",
                "max_seq_lens",
                "teacher_log_probs",
                "triage_token_weight_units",  # {MARKER}
'''
    text = _replace_once(text, old, new, path=path)
    path.write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--slime-root", type=Path, default=DEFAULT_SLIME_ROOT)
    args = parser.parse_args()

    targets = {relative: args.slime_root / relative for relative in STOCK_SHA256}
    for relative, path in targets.items():
        if not path.is_file() or path.is_symlink():
            raise SystemExit(f"TOKENMEAN_PATCH_REFUSED missing regular file: {path}")
        text = path.read_text(encoding="utf-8")
        if MARKER not in text:
            actual = _sha256(path)
            if actual != STOCK_SHA256[relative]:
                raise SystemExit(
                    f"TOKENMEAN_PATCH_REFUSED stock hash mismatch for {path}: {actual}"
                )

    patch_loss(targets["slime/backends/megatron_utils/loss.py"])
    patch_model(targets["slime/backends/megatron_utils/model.py"])
    for path in targets.values():
        if MARKER not in path.read_text(encoding="utf-8"):
            raise SystemExit(f"TOKENMEAN_PATCH_REFUSED marker missing after patch: {path}")
    print(
        "SLIME_TOKENMEAN_CUSTOM_NORMALIZER_PATCH_OK "
        f"slime_root={args.slime_root} files={len(targets)}"
    )


if __name__ == "__main__":
    main()
