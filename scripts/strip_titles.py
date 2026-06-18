#!/usr/bin/env python3
"""Remove the redundant Plotly config titles from the batch-generated figures.

Each figure embeds a title such as
``SKCM | TC | offaug_AE_head_2 | CVAE1-50 — f1_score`` both as the layout title
and inside the metric dropdown (``updatemenus``) button arguments, so switching
metric re-applies it. The website already shows this information in its caption,
so we blank every such title. Axis titles ("Sample size") are left untouched
because they contain no em dash.

Run standalone to clean every figure under ``plots/``:

    python3 scripts/strip_titles.py

or import :func:`strip_title` from other scripts (e.g. the migrator).
"""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PLOTS_ROOT = ROOT / "plots"

# The config titles always contain an em dash (raw "—" or escaped "\u2014");
# axis titles do not, so this distinguishes them safely.
_EMDASH = r"(?:\u2014|\\u2014)"
# Object form:  "title":{"text":"... — metric"
_OBJ_RE = re.compile(r'("title":\{"text":")[^"]*' + _EMDASH + r'[^"]*(")')
# String form (dropdown button args):  "title":"... — metric"
_STR_RE = re.compile(r'("title":")[^"]*' + _EMDASH + r'[^"]*(")')


def strip_title(html: str) -> str:
    html = _OBJ_RE.sub(r"\1\2", html)
    html = _STR_RE.sub(r"\1\2", html)
    return html


def main() -> None:
    changed = 0
    for path in sorted(PLOTS_ROOT.rglob("learning_curve.html")):
        text = path.read_text(encoding="utf-8")
        new = strip_title(text)
        if new != text:
            path.write_text(new, encoding="utf-8")
            changed += 1
    print(f"Stripped titles in {changed} file(s).")


if __name__ == "__main__":
    main()
