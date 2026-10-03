"""Unit tests for the LaTeX helpers behind the reproducibility sheet."""

from __future__ import annotations

import unittest

from aws.reporting.latex import Raw, fmt, latex, render, table, verbatim


class EscapeTests(unittest.TestCase):
    """Verifies special and non-ASCII characters are made safe."""

    def test_special_characters_are_escaped(self) -> None:
        self.assertEqual(latex("50% of $5 & #1_a"), r"50\% of \$5 \& \#1\_a")

    def test_known_unicode_is_replaced_and_unknown_fails(self) -> None:
        self.assertEqual(latex("0.578 → 0.781 ≈ ×2"), r"0.578 $\rightarrow$ 0.781 $\approx$ $\times$2")
        with self.assertRaises(ValueError):
            latex("emoji 🙂")


class FormatTests(unittest.TestCase):
    """Verifies cells are formatted consistently."""

    def test_numbers_missing_values_and_raw_cells(self) -> None:
        self.assertEqual(fmt(0.578125), "0.5781")
        self.assertEqual(fmt(2.0), "2")
        self.assertEqual(fmt(4833702), "4,833,702")
        self.assertEqual(fmt(None), "--")
        self.assertEqual(fmt(float("nan")), "--")
        self.assertEqual(fmt(True), "yes")
        self.assertEqual(fmt(Raw(r"\texttt{x}")), r"\texttt{x}")
        self.assertEqual(fmt("a_b"), r"a\_b")


class TableTests(unittest.TestCase):
    """Verifies table shapes and the page-breaking variant."""

    def test_rows_must_match_headers(self) -> None:
        with self.assertRaises(ValueError):
            table(["a", "b"], [[1]], "ll")

    def test_short_and_long_tables(self) -> None:
        short = table(["Metric", "Value"], [["pass@1", 0.5]], "lr", caption="Results")
        long = table(["Step"], [[1], [2]], "r", caption="Steps", long=True)

        self.assertIn(r"\captionof{table}{Results}", short)
        self.assertIn(r"pass@1 & 0.5 \\", short)
        self.assertIn(r"\begin{longtable}{r}", long)
        self.assertIn(r"\endhead", long)

    def test_verbatim_keeps_commands_as_typed(self) -> None:
        self.assertIn("python scripts/14_verify_invariants.py --x", verbatim(["python scripts/14_verify_invariants.py --x"]))


class RenderTests(unittest.TestCase):
    """Verifies every placeholder must be filled."""

    def test_placeholders_are_filled_and_missing_ones_fail(self) -> None:
        self.assertEqual(render("a <<ONE>> b", {"ONE": "1"}), "a 1 b")
        with self.assertRaisesRegex(KeyError, "TWO"):
            render("<<ONE>> <<TWO>>", {"ONE": "1"})


if __name__ == "__main__":
    unittest.main()
