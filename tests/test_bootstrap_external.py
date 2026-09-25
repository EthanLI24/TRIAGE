#!/usr/bin/env python3
"""bootstrap_external.sh routes every public manifest entry to git and prints
the NOTE for the internal miniTransformer release, without collapsing empty
manifest fields."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]

GIT_STUB = """#!/usr/bin/env bash
{
  printf 'CALL'
  for arg in "$@"; do printf '\\t%s' "$arg"; done
  printf '\\n'
} >> "$TRIAGE_TEST_GIT_LOG"
"""

MKDIR_STUB = "#!/usr/bin/env bash\n"


class BootstrapExternalTest(unittest.TestCase):
    def test_manifest_rows_route_to_git_or_note(self):
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory)
            (work / "scripts").mkdir()
            shutil.copy(ROOT / "scripts" / "bootstrap_external.sh", work / "scripts" / "bootstrap_external.sh")
            os.symlink(ROOT / "compatibility.json", work / "compatibility.json")
            shim = work / "shim"
            shim.mkdir()
            for name, text in (("git", GIT_STUB), ("mkdir", MKDIR_STUB)):
                stub = shim / name
                stub.write_text(text)
                stub.chmod(0o755)
            env = dict(os.environ)
            env["PATH"] = f"{shim}{os.pathsep}{env.get('PATH', '')}"
            env["TRIAGE_TEST_GIT_LOG"] = str(work / "git.log")
            result = subprocess.run(["bash", str(work / "scripts" / "bootstrap_external.sh")],
                                    env=env, text=True, capture_output=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(result.stderr, "")

            manifest = json.loads((ROOT / "compatibility.json").read_text())["reference_stack"]
            self.assertIsNone(manifest["mini_transformer"]["url"], "manifest must keep mini_transformer internal")

            calls = [line.split("\t")[1:] for line in (work / "git.log").read_text().splitlines()]
            expected = []
            for key, dirname in (("slime", "slime"), ("sglang", "sglang"), ("megatron_lm", "Megatron-LM")):
                entry = manifest[key]
                dest = os.path.join(str(work), "external", dirname)
                expected.append(["clone", entry["url"], dest])
                expected.append(["-C", dest, "checkout", entry["commit"]])
            self.assertEqual(calls, expected)

            routed = [line for line in result.stdout.splitlines() if line.startswith(("Cloning ", "NOTE: "))]
            self.assertEqual(len(routed), 4, "all four manifest rows must be processed")
            self.assertIn("NOTE: mini_transformer is an internal release", result.stdout)
            self.assertIn(manifest["mini_transformer"]["commit"], result.stdout)
            self.assertIn(os.path.join("external", "miniTransformer"), result.stdout)


if __name__ == "__main__":
    unittest.main()
