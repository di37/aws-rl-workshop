"""Small LaTeX helpers: escape text, format cells, build tables, and fill a template.

Used to build the reproducibility sheet from evidence, so no number in it is
typed by hand.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any

_PLACEHOLDER = re.compile(r"<<([A-Z0-9_]+)>>")
_SPECIAL = {
    "\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#", "_": r"\_",
    "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}", "^": r"\textasciicircum{}",
}
_UNICODE = {
    "→": r"$\rightarrow$", "≥": r"$\geq$", "≤": r"$\leq$", "×": r"$\times$", "≈": r"$\approx$",
    "±": r"$\pm$", "–": "--", "—": "---", "…": r"\ldots{}", "’": "'", "‘": "`", "“": "``", "”": "''",
}


class Raw(str):
    """A table cell or value that is already LaTeX and must not be escaped."""


def latex(text: Any) -> str:
    """Escapes text for LaTeX.

    Args:
        text: Any value; converted with ``str``.

    Returns:
        Text safe to place in a LaTeX document.

    Raises:
        ValueError: If the text contains a non-ASCII character with no known replacement.
    """
    out = []
    for char in str(text):
        if char in _SPECIAL:
            out.append(_SPECIAL[char])
        elif char in _UNICODE:
            out.append(_UNICODE[char])
        elif ord(char) < 128:
            out.append(char)
        else:
            raise ValueError(f"No LaTeX replacement for {char!r} in {str(text)[:60]!r}.")
    return "".join(out)


def fmt(value: Any, digits: int = 4) -> str:
    """Formats one table cell.

    Args:
        value: Cell value.
        digits: Decimal places kept for floats (trailing zeros dropped).

    Returns:
        LaTeX text: ``--`` for missing values, grouped digits for integers,
        rounded floats, and escaped text otherwise.
    """
    if isinstance(value, Raw):
        return value
    if value is None or (isinstance(value, float) and value != value) or value == "":
        return "--"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, int):
        return f"{value:,}"
    if isinstance(value, float):
        text = f"{value:.{digits}f}".rstrip("0").rstrip(".")
        return text if text not in ("", "-0") else "0"
    return latex(value)


def table(
    headers: Sequence[str],
    rows: Sequence[Sequence[Any]],
    spec: str,
    caption: str | None = None,
    long: bool = False,
) -> str:
    """Builds a booktabs table; ``long`` makes it break across pages with a repeated header.

    Args:
        headers: Column headers (escaped).
        rows: Cell values, formatted with :func:`fmt`.
        spec: LaTeX column specification, such as ``l r p{6cm}``.
        caption: Optional caption (already LaTeX).
        long: Whether to use ``longtable``.

    Returns:
        The LaTeX table.

    Raises:
        ValueError: If a row's length differs from the header's.
    """
    if any(len(row) != len(headers) for row in rows):
        raise ValueError("Every row must have one cell per header.")
    head = " & ".join(rf"\textbf{{{latex(h)}}}" for h in headers) + r" \\"
    body = "\n".join(" & ".join(fmt(cell) for cell in row) + r" \\" for row in rows)
    if long:
        continued = rf"\multicolumn{{{len(headers)}}}{{l}}{{\textit{{continued}}}} \\"
        title = rf"\caption{{{caption}}} \\" if caption else ""
        return (f"\\begin{{longtable}}{{{spec}}}\n{title}\n\\toprule\n{head}\n\\midrule\n\\endfirsthead\n"
                f"{continued}\n\\toprule\n{head}\n\\midrule\n\\endhead\n\\bottomrule\n\\endlastfoot\n"
                f"{body}\n\\end{{longtable}}")
    title = f"\\captionof{{table}}{{{caption}}}\n" if caption else ""
    return (f"\\begin{{center}}\n\\begin{{minipage}}{{\\linewidth}}\\centering\n{title}"
            f"\\begin{{tabular}}{{{spec}}}\n\\toprule\n{head}\n\\midrule\n{body}\n"
            f"\\bottomrule\n\\end{{tabular}}\n\\end{{minipage}}\n\\end{{center}}")


def breakable(text: str) -> str:
    """Typewriter text that may break across lines (paths, ARNs, and URIs)."""
    return rf"\nolinkurl{{{text}}}"


def verbatim(lines: Sequence[str]) -> str:
    """Wraps command lines in a framed verbatim block (no escaping needed inside)."""
    return "\\begin{Verbatim}[frame=single,fontsize=\\small]\n" + "\n".join(lines) + "\n\\end{Verbatim}"


def render(template: str, values: Mapping[str, str]) -> str:
    """Fills every ``<<KEY>>`` placeholder in a template.

    Args:
        template: LaTeX text with placeholders.
        values: Placeholder name to LaTeX text.

    Returns:
        The filled document.

    Raises:
        KeyError: If a placeholder has no value.
    """
    missing = sorted(set(_PLACEHOLDER.findall(template)) - set(values))
    if missing:
        raise KeyError(f"No value for placeholders: {missing}")
    return _PLACEHOLDER.sub(lambda match: values[match.group(1)], template)
