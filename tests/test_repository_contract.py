#!/usr/bin/env python3
"""Static repository contract tests that do not require Slime or a GPU."""

from __future__ import annotations

import hashlib
import json
import subprocess
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

RESULTS_START = "<!-- BEGIN GENERATED RESULTS -->"
RESULTS_END = "<!-- END GENERATED RESULTS -->"

EXPECTED_SLIME_PATCHES = (
    "patch_routing_replay_indices.py",
    "patch_slime_r3_route_validation.py",
    "patch_tokenmean_custom_normalizer.py",
)
EXPECTED_SGLANG_PATCHES = (
    "patch_sglang_r3_bypassed_topk.py",
    "patch_sglang_r3_force_return.py",
)

# Writing-rule tokens that must not appear in the user-facing README.
FORBIDDEN_README_TOKENS = ("TBD", "ICLR2027", "finals/Latex", "manuscript")


class RepositoryContractTest(unittest.TestCase):
    def test_overlay_hashes_match_manifest(self) -> None:
        manifest = json.loads((ROOT / "compatibility.json").read_text())
        for relative, expected in manifest["overlay_sha256"].items():
            actual = hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()
            self.assertEqual(actual, expected, relative)

    def test_shell_scripts_parse(self) -> None:
        scripts = [p for p in ROOT.rglob("*.sh") if ".venv" not in p.parts]
        scripts += list((ROOT / "configs").glob("*.env"))
        for relative in scripts:
            subprocess.run(
                ["bash", "-n", str(ROOT / relative)],
                check=True,
                capture_output=True,
                text=True,
            )

    def test_license_exists(self) -> None:
        license_text = (ROOT / "LICENSE").read_text()
        self.assertIn("Apache License", license_text)
        self.assertIn("Copyright 2026 The TRIAGE Authors", license_text)

    def test_readme_layout(self) -> None:
        self.assertTrue((ROOT / "README.md").is_file(), "README.md missing")
        self.assertTrue((ROOT / "README_zh.md").is_file(), "README_zh.md missing")
        self.assertFalse((ROOT / "README_CN.md").exists(), "README_CN.md must be renamed to README_zh.md")

    def test_validation_doc_lives_in_docs(self) -> None:
        self.assertTrue((ROOT / "docs" / "VALIDATION.md").is_file(), "docs/VALIDATION.md missing")
        self.assertFalse((ROOT / "VALIDATION.md").exists(), "VALIDATION.md must live under docs/")

    def test_generated_results_markers_in_benchmarks_only(self) -> None:
        benchmarks = (ROOT / "benchmarks" / "README.md").read_text()
        self.assertIn(RESULTS_START, benchmarks)
        self.assertIn(RESULTS_END, benchmarks)
        readme = (ROOT / "README.md").read_text()
        self.assertNotIn(RESULTS_START, readme)
        self.assertNotIn(RESULTS_END, readme)

    def test_patch_directories(self) -> None:
        for name in EXPECTED_SLIME_PATCHES:
            self.assertTrue((ROOT / "patches" / "slime" / name).is_file(), f"patches/slime/{name} missing")
        for name in EXPECTED_SGLANG_PATCHES:
            self.assertTrue((ROOT / "patches" / "sglang" / name).is_file(), f"patches/sglang/{name} missing")

    def test_readme_writing_rules(self) -> None:
        readme = (ROOT / "README.md").read_text().casefold()
        for token in FORBIDDEN_README_TOKENS:
            self.assertNotIn(token.casefold(), readme, f"README.md must not contain {token!r}")


if __name__ == "__main__":
    unittest.main()
