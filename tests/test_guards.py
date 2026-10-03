"""Unit tests for the shared billable-operation confirmation check."""

from __future__ import annotations

import unittest

from aws.guards import require_phrase


class RequirePhraseTests(unittest.TestCase):
    """Verifies exact-match confirmation."""

    def test_exact_phrase_passes(self) -> None:
        require_phrase("CONFIRM_X", "CONFIRM_X", "training")

    def test_wrong_phrase_names_action_and_expected_phrase(self) -> None:
        with self.assertRaisesRegex(PermissionError, "Billable training blocked.*CONFIRM_X"):
            require_phrase("confirm_x", "CONFIRM_X", "training")


if __name__ == "__main__":
    unittest.main()
