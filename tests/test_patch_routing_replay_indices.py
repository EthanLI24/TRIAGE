#!/usr/bin/env python3
"""CPU-only regression tests for the Slime routing-index patcher."""

from __future__ import annotations

import importlib.util
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path


PATCHER_PATH = (
    Path(__file__).resolve().parents[1]
    / "patches"
    / "slime"
    / "patch_routing_replay_indices.py"
)
SPEC = importlib.util.spec_from_file_location("routing_index_patcher", PATCHER_PATH)
assert SPEC is not None and SPEC.loader is not None
PATCHER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PATCHER)


class PatchRoutingReplayIndicesTest(unittest.TestCase):
    def make_target(self, text: str) -> Path:
        tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False)
        self.addCleanup(Path(tmp.name).unlink, missing_ok=True)
        tmp.write(text)
        tmp.close()
        return Path(tmp.name)

    def run_main(self, target: Path) -> None:
        previous = PATCHER.TARGET
        PATCHER.TARGET = target
        try:
            with redirect_stdout(StringIO()):
                PATCHER.main()
        finally:
            PATCHER.TARGET = previous

    def test_applies_twice_idempotently(self) -> None:
        target = self.make_target(
            "prefix\n"
            + PATCHER.OLD
            + "\nmiddle\n"
            + PATCHER.OLD
            + "\nsuffix\n"
        )
        self.run_main(target)
        self.run_main(target)
        source = target.read_text()
        self.assertEqual(source.count(PATCHER.NEW), 2)
        self.assertNotIn(PATCHER.OLD, source)

    def test_fails_closed_on_unknown_source(self) -> None:
        target = self.make_target("# unexpected Slime revision\n")
        with self.assertRaisesRegex(RuntimeError, "unexpected source"):
            self.run_main(target)


if __name__ == "__main__":
    unittest.main()
