"""Unit tests for the provenance helpers (what actually ran)."""

from __future__ import annotations

import tempfile
import unittest
import zipfile
from pathlib import Path

from aws.records.provenance import (
    FIXED_IMPORT_FILES,
    compare_copies,
    module_constant,
    parse_build_log,
    zip_hashes,
)

BUILD_LOG = """#4 [1/6] FROM docker.io/library/python:3.12-slim@sha256:{base}
#9 55.72 Successfully installed Flask-3.1.3 aws-bedrock-token-generator-1.1.0 bedrock-agentcore-1.24.0
#11 writing image sha256:{other} done
20261001-231128-336: digest: sha256:{digest} size: 2202
""".format(base="a" * 64, other="b" * 64, digest="c" * 64)


class BuildLogTests(unittest.TestCase):
    """Verifies the image digests and installed packages are read from a CodeBuild log."""

    def test_base_image_pushed_digest_and_packages_are_parsed(self) -> None:
        parsed = parse_build_log(BUILD_LOG, image_tag="20261001-231128-336")

        self.assertEqual(parsed["base_image"], "python:3.12-slim@sha256:" + "a" * 64)
        self.assertEqual(parsed["image_digest"], "sha256:" + "c" * 64)
        self.assertEqual(parsed["packages"], {"aws-bedrock-token-generator": "1.1.0",
                                              "bedrock-agentcore": "1.24.0", "flask": "3.1.3"})

    def test_log_of_another_image_has_no_digest(self) -> None:
        self.assertIsNone(parse_build_log(BUILD_LOG, image_tag="another-tag")["image_digest"])


class SourceBundleTests(unittest.TestCase):
    """Verifies bundle hashing and constant extraction from source text."""

    def test_zip_hashes_cover_every_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "source.zip"
            with zipfile.ZipFile(path, "w") as archive:
                archive.writestr("app.py", "x = 1\n")
                archive.writestr("nested/", "")

            hashes = zip_hashes(path)

        self.assertEqual(list(hashes), ["app.py"])
        self.assertEqual(len(hashes["app.py"]), 64)

    def test_module_constant_reads_a_string_assignment(self) -> None:
        source = 'import x\nSYSTEM_PROMPT = """Fix the ticket."""\nOTHER = 2\n'

        self.assertEqual(module_constant(source, "SYSTEM_PROMPT"), "Fix the ticket.")
        self.assertIsNone(module_constant(source, "MISSING"))


class CopyComparisonTests(unittest.TestCase):
    """Verifies each copied file gets an explicit verdict."""

    def test_verdicts_distinguish_identical_resized_fixed_and_missing(self) -> None:
        source = {"a.safetensors": (10, "e1"), "config.json": (5, "e2"), "tokenizer_config.json": (4, "e3"),
                  "version": (1, "e4"), "b.safetensors": (7, "e5")}
        copy = {"a.safetensors": (10, "other"), "config.json": (5, "e2"), "tokenizer_config.json": (9, "e9"),
                "b.safetensors": (8, "e6")}

        verdicts = {row["file"]: row["verdict"] for row in compare_copies(source, copy, FIXED_IMPORT_FILES)}

        self.assertEqual(verdicts["config.json"], "identical")
        self.assertEqual(verdicts["a.safetensors"], "same size")
        self.assertEqual(verdicts["tokenizer_config.json"], "changed on purpose")
        self.assertEqual(verdicts["version"], "not copied")
        self.assertEqual(verdicts["b.safetensors"], "DIFFERENT")


if __name__ == "__main__":
    unittest.main()
