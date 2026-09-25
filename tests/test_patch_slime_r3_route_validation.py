#!/usr/bin/env python3
"""CPU-only patcher tests for R3 route validation."""

from __future__ import annotations

import importlib.util
import tempfile
import types
import unittest
from pathlib import Path


PATCHER_PATH = (
    Path(__file__).resolve().parents[1]
    / "patches"
    / "slime"
    / "patch_slime_r3_route_validation.py"
)
SPEC = importlib.util.spec_from_file_location("r3_route_validation_patcher", PATCHER_PATH)
assert SPEC is not None and SPEC.loader is not None
PATCHER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PATCHER)


class PatchSlimeR3RouteValidationTest(unittest.TestCase):
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
        self.assertIn("args.use_rollout_routing_replay", patched)
        self.assertIn("SGLang response is missing routed_experts", patched)
        self.assertIn("np.unique(route_probe).size", patched)
        self.assertIn("Invalid rollout routed experts", patched)

    def make_runtime_function(self):
        source = (
            "import base64 as pybase64\n"
            "import numpy as np\n"
            "def decode_routes(args, output, sample):\n"
            + PATCHER.OLD
            + "    return sample\n"
        )
        target = self.make_target(source)
        self.assertEqual(PATCHER.patch(target), "applied")
        namespace: dict[str, object] = {}
        exec(compile(target.read_text(), str(target), "exec"), namespace)
        return namespace["decode_routes"]

    def test_missing_route_key_fails_immediately_when_r3_enabled(self) -> None:
        decode_routes = self.make_runtime_function()
        args = types.SimpleNamespace(
            use_rollout_routing_replay=True,
            num_layers=2,
            moe_router_topk=2,
            num_experts=8,
        )
        sample = types.SimpleNamespace(tokens=[1, 2])

        with self.assertRaisesRegex(ValueError, "missing routed_experts"):
            decode_routes(args, {"meta_info": {}}, sample)

    def test_missing_route_key_is_ignored_when_r3_disabled(self) -> None:
        decode_routes = self.make_runtime_function()
        args = types.SimpleNamespace(
            use_rollout_routing_replay=False,
            num_layers=2,
            moe_router_topk=2,
            num_experts=8,
        )
        sample = types.SimpleNamespace(tokens=[1, 2])

        self.assertIs(decode_routes(args, {"meta_info": {}}, sample), sample)

    def test_fails_closed_on_unknown_source(self) -> None:
        target = self.make_target("# unexpected Slime revision\n")
        with self.assertRaisesRegex(RuntimeError, "unexpected source"):
            PATCHER.patch(target)


if __name__ == "__main__":
    unittest.main()
