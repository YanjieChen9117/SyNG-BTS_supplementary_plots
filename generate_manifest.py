#!/usr/bin/env python3
"""Scan the data directory and build manifest.json used by the website.

Current expected layout (one plot per leaf directory):

    data/learning_curve_output/cohorts/<group>/<subtype>/<normalization>/offaug_<offaug>/<param>/learning_curve.html

where <group> is parsed as ``<cohort>_<datatype>_<model>`` (split on "_",
first token = cohort, last token = model, middle = data type).

Running this script regenerates ``manifest.json`` at the repo root. Re-run it
whenever plots are added or removed.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PLOTS_ROOT = ROOT / "data" / "learning_curve_output" / "cohorts"
PLOT_FILENAME = "learning_curve.html"
OUTPUT = ROOT / "manifest.json"

# Pretty labels for known tokens. Unknown tokens fall back to a sensible default.
DATA_TYPE_LABELS = {
    "rna": "RNA",
    "mrna": "mRNA",
    "mirna": "miRNA",
    "methy": "Methylation",
    "methylation": "Methylation",
    "protein": "Protein",
    "cnv": "CNV",
}


def parse_group(group: str) -> dict[str, str]:
    """Parse a group folder name like ``fivesubtypes_rna_cvae``."""
    tokens = group.split("_")
    if len(tokens) >= 3:
        cohort = tokens[0]
        model = tokens[-1]
        data_token = "_".join(tokens[1:-1])
    elif len(tokens) == 2:
        cohort, data_token, model = tokens[0], tokens[1], ""
    else:
        cohort, data_token, model = group, "", ""

    data_type = DATA_TYPE_LABELS.get(data_token.lower(), data_token.upper() or "Unknown")
    return {
        "cohort": cohort,
        "data_type": data_type,
        "model": model.upper(),
    }


def parse_offaug(name: str) -> str:
    """``offaug_none`` -> ``None``; ``offaug_AE_head_2`` -> ``AE_head_2``."""
    value = name[len("offaug_"):] if name.startswith("offaug_") else name
    return "None" if value.lower() == "none" else value


def build_manifest() -> dict:
    plots: list[dict] = []

    if not PLOTS_ROOT.is_dir():
        raise SystemExit(f"Plots root not found: {PLOTS_ROOT}")

    for html_path in sorted(PLOTS_ROOT.rglob(PLOT_FILENAME)):
        rel = html_path.relative_to(PLOTS_ROOT)
        parts = rel.parts
        # expected: <group>/<subtype>/<norm>/offaug_<x>/<param>/learning_curve.html
        if len(parts) != 6:
            print(f"  [skip] unexpected depth: {rel}")
            continue

        group, subtype, norm, offaug_dir, param, _ = parts
        meta = parse_group(group)

        plots.append(
            {
                "data_type": meta["data_type"],
                "subtype": subtype,
                "normalization": norm,
                "offaug": parse_offaug(offaug_dir),
                "param": param,
                "cohort": meta["cohort"],
                "model": meta["model"],
                "path": str(html_path.relative_to(ROOT)).replace("\\", "/"),
            }
        )

    return {
        "plot_type": "learning_curve",
        "count": len(plots),
        # Order in which selectors are rendered on the site.
        "dimensions": [
            {"key": "data_type", "label": "Data type"},
            {"key": "subtype", "label": "Cancer subtype"},
            {"key": "normalization", "label": "Normalization"},
            {"key": "param", "label": "Parameters"},
            {"key": "offaug", "label": "Offline augmentation"},
        ],
        "plots": plots,
    }


def main() -> None:
    manifest = build_manifest()
    OUTPUT.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {OUTPUT.relative_to(ROOT)} with {manifest['count']} plots.")


if __name__ == "__main__":
    main()
