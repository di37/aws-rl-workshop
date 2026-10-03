"""Shared confirmation check for billable operations."""

from __future__ import annotations


def require_phrase(actual: str, expected: str, action: str) -> None:
    """Allows a billable operation only with its exact confirmation phrase.

    Args:
        actual: Phrase supplied by the user.
        expected: Exact phrase required for the operation.
        action: Operation name used in the error, such as ``training``.

    Raises:
        PermissionError: If the phrases differ.
    """
    if actual != expected:
        raise PermissionError(f"Billable {action} blocked. Required confirmation: {expected}")
