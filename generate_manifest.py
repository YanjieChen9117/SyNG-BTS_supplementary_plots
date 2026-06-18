#!/usr/bin/env python3
"""Scan the ``plots/`` directory and build ``manifest.json`` for the website.

Clean unified layout (one figure per leaf directory):

    plots/<data_type>/<subtype>/<group_label>/<normalization>/<offaug>/<param>/learning_curve.html

- ``data_type``    e.g. RNA, miRNA
- ``subtype``      cancer code, e.g. SKCM, COAD, KIRP
- ``group_label``  classification target (shown as caption, not a selector)
- ``normalization``raw / TC / DESeq
- ``offaug``       offline augmentation: none / AE_head_2 / ...
- ``param``        model config, e.g. CVAE1-50

Run this whenever plots are added or removed:

    python3 generate_manifest.py
"""
from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PLOTS_ROOT = ROOT / "plots"
PLOT_FILENAME = "learning_curve.html"
OUTPUT = ROOT / "manifest.json"
SAMPLE_TSV = ROOT / "sample_size.tsv"

# The five selector dimensions shown on the site, in display order.
DIMENSIONS = [
    {"key": "data_type", "label": "Data type"},
    {"key": "subtype", "label": "Cancer subtype"},
    {"key": "normalization", "label": "Normalization method"},
    {"key": "param", "label": "Augmentation model parameters"},
    {"key": "offaug", "label": "Offline augmentation"},
]

# Preferred ordering for dimensions with a natural (non-alphabetical) order.
PREFERRED_ORDER = {
    "data_type": ["miRNA", "RNA"],
    "normalization": ["raw", "TC", "DESeq"],
    "offaug": ["AE", "none"],
}


def load_dataset_stats() -> dict[tuple[str, str, str], dict[str, str]]:
    """Map (data_type, subtype, group_label) -> sample size & marker dimension.

    Source columns from ``sample_size.tsv``: ``N (after NA)`` and
    ``Markers (final)``. Rows prefixed with ``RNA_`` are RNA datasets; the rest
    are miRNA.
    """
    stats: dict[tuple[str, str, str], dict[str, str]] = {}
    if not SAMPLE_TSV.exists():
        print(f"  [warn] {SAMPLE_TSV.name} not found; dataset stats omitted.")
        return stats

    lines = SAMPLE_TSV.read_text(encoding="utf-8").splitlines()
    if not lines:
        return stats
    header = lines[0].split("\t")
    try:
        n_idx = header.index("N (after NA)")
        m_idx = header.index("Markers (final)")
    except ValueError:
        print("  [warn] expected columns missing in sample_size.tsv.")
        return stats

    for line in lines[1:]:
        if not line.strip():
            continue
        cols = line.split("\t")
        name = cols[0].strip()
        if name.startswith("RNA_"):
            data_type, rest = "RNA", name[len("RNA_"):]
        else:
            data_type, rest = "miRNA", name
        subtype, _, group_label = rest.partition("_")
        stats[(data_type, subtype, group_label)] = {
            "sample_size": cols[n_idx].strip(),
            "marker_dim": cols[m_idx].strip(),
        }
    return stats


def order_values(key: str, values: set[str]) -> list[str]:
    vals = list(values)
    if key == "param":
        # Sort numerically by the trailing integer (CVAE1-5 < CVAE1-10 < CVAE1-50).
        def num(v: str) -> tuple:
            m = re.search(r"(\d+)\s*$", v)
            return (int(m.group(1)) if m else 0, v)
        return sorted(vals, key=num)
    pref = PREFERRED_ORDER.get(key)
    if pref:
        rank = {v: i for i, v in enumerate(pref)}
        return sorted(vals, key=lambda v: (rank.get(v, len(pref)), v))
    return sorted(vals)


def build_manifest() -> dict:
    if not PLOTS_ROOT.is_dir():
        raise SystemExit(f"Plots root not found: {PLOTS_ROOT}")

    stats = load_dataset_stats()

    plots: list[dict] = []
    missing_stats: set[tuple[str, str, str]] = set()
    for html_path in sorted(PLOTS_ROOT.rglob(PLOT_FILENAME)):
        rel = html_path.relative_to(PLOTS_ROOT).parts
        # <data_type>/<subtype>/<group_label>/<norm>/<offaug>/<param>/learning_curve.html
        if len(rel) != 7:
            print(f"  [skip] unexpected depth: {html_path.relative_to(ROOT)}")
            continue
        data_type, subtype, group_label, norm, offaug, param, _ = rel
        stat = stats.get((data_type, subtype, group_label))
        if stat is None:
            missing_stats.add((data_type, subtype, group_label))
        plots.append(
            {
                "data_type": data_type,
                "subtype": subtype,
                "normalization": norm,
                "param": param,
                "offaug": offaug,
                "group_label": group_label,
                "sample_size": stat["sample_size"] if stat else None,
                "marker_dim": stat["marker_dim"] if stat else None,
                "path": str(html_path.relative_to(ROOT)).replace("\\", "/"),
            }
        )

    for key in sorted(missing_stats):
        print(f"  [warn] no sample_size.tsv row for {key}")

    # Pre-compute ordered value lists per selector so the UI renders consistently.
    dimensions = []
    for dim in DIMENSIONS:
        values = order_values(dim["key"], {p[dim["key"]] for p in plots})
        dimensions.append({**dim, "values": values})

    return {
        "plot_type": "learning_curve",
        "count": len(plots),
        "dimensions": dimensions,
        "plots": plots,
    }


def main() -> None:
    manifest = build_manifest()
    OUTPUT.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {OUTPUT.relative_to(ROOT)} with {manifest['count']} plots.")
    for dim in manifest["dimensions"]:
        print(f"  {dim['label']}: {dim['values']}")


if __name__ == "__main__":
    main()
