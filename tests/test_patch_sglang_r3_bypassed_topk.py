#!/usr/bin/env python3
"""CPU-only regression tests for the SGLang R3 BYPASSED Top-K patcher."""

from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path


PATCHER_PATH = (
    Path(__file__).resolve().parents[1]
    / "patches"
    / "sglang"
    / "patch_sglang_r3_bypassed_topk.py"
)
SPEC = importlib.util.spec_from_file_location("r3_bypassed_topk_patcher", PATCHER_PATH)
assert SPEC is not None and SPEC.loader is not None
PATCHER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PATCHER)


class PatchSglangR3BypassedTopkTest(unittest.TestCase):
    def make_target(self, text: str) -> Path:
        tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False)
        self.addCleanup(Path(tmp.name).unlink, missing_ok=True)
        tmp.write(text)
        tmp.close()
        return Path(tmp.name)

    def test_applies_once_and_is_idempotent(self) -> None:
        target = self.make_target("prefix\n" + PATCHER.OLD + "suffix\n")
        self.assertEqual(PATCHER.patch(target), "applied")
        self.assertEqual(PATCHER.patch(target), "already applied")
        patched = target.read_text()
        self.assertNotIn(PATCHER.OLD, patched)
        self.assertEqual(patched.count(PATCHER.NEW), 1)

        # The side channel is gated, invokes the existing capture-bearing
        # select_experts path, and still returns BypassedTopKOutput.
        self.assertIn("enable_return_routed_experts", patched)
        self.assertIn("select_experts(", patched)
        self.assertIn("return BypassedTopKOutput(", patched)

    def test_fails_closed_on_unknown_source(self) -> None:
        target = self.make_target("# unexpected SGLang revision\n")
        with self.assertRaisesRegex(RuntimeError, "unexpected source"):
            PATCHER.patch(target)


if __name__ == "__main__":
    unittest.main()
