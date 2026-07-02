#!/usr/bin/env python3
"""One-off migration: copy the messy batch outputs into the clean ``plots/`` tree.

Source batches (different layouts):

  learning_curve_output_full_cohort/cohorts/fivesubtypes_rna_cvae_full_cohort/
      <subtype>/<norm>/offaug_<x>/<param>/learning_curve.html        (has offaug)

  data/learning_curve_output_miRNA/
      <CANCER>_<group_label>/<norm>/<param>/learning_curve.html      (no offaug)

Unified target (self-describing, uniform depth):

  plots/<data_type>/<subtype>/<group_label>/<norm>/<offaug>/<param>/learning_curve.html

Only ``learning_curve.html`` is copied (the website needs nothing else).
This script is idempotent: re-running overwrites existing files.
"""
from __future__ import annotations

from pathlib import Path

from strip_titles import strip_title

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
DEST_ROOT = ROOT / "plots"
PLOT_FILE = "learning_curve.html"

# Group label (classification target) for the RNA batch, by cancer subtype.
RNA_GROUP_LABELS = {
    "COAD": "tumor_status",
    "SKCM": "breslow_thickness_at_diagnosis",
}

# Normalize offline-augmentation folder names to the labels shown on the site.
OFFAUG_LABELS = {
    "none": "none",
    "AE_head_2": "AE",
}

copied = 0
skipped: list[str] = []


def copy_one(src: Path, data_type: str, subtype: str, group_label: str,
             norm: str, offaug: str, param: str) -> None:
    global copied
    offaug = OFFAUG_LABELS.get(offaug, offaug)
    dest = DEST_ROOT / data_type / subtype / group_label / norm / offaug / param / PLOT_FILE
    dest.parent.mkdir(parents=True, exist_ok=True)
    # Copy while removing the redundant embedded figure title.
    dest.write_text(strip_title(src.read_text(encoding="utf-8")), encoding="utf-8")
    copied += 1


def migrate_rna() -> None:
    base = ROOT / "learning_curve_output_full_cohort" / "cohorts" / "fivesubtypes_rna_cvae_full_cohort"
    if not base.is_dir():
        print(f"[RNA] base not found, skipping: {base}")
        return
    for html in sorted(base.rglob(PLOT_FILE)):
        rel = html.relative_to(base).parts  # <subtype>/<norm>/offaug_<x>/<param>/learning_curve.html
        if len(rel) != 5:
            skipped.append(str(html))
            continue
        subtype, norm, offaug_dir, param, _ = rel
        if subtype not in RNA_GROUP_LABELS:
            skipped.append(str(html))
            continue
        offaug = offaug_dir[len("offaug_"):] if offaug_dir.startswith("offaug_") else offaug_dir
        group_label = RNA_GROUP_LABELS[subtype]
        copy_one(html, "RNA", subtype, group_label, norm, offaug, param)


def migrate_mirna() -> None:
    base = DATA / "learning_curve_output_miRNA"
    if not base.is_dir():
        print(f"[miRNA] base not found, skipping: {base}")
        return
    for html in sorted(base.rglob(PLOT_FILE)):
        rel = html.relative_to(base).parts  # <CANCER_label>/<norm>/<param>/learning_curve.html
        if len(rel) != 4:
            skipped.append(str(html))
            continue
        cohort, norm, param, _ = rel
        cancer, _, group_label = cohort.partition("_")  # split on first underscore
        group_label = group_label or "unknown"
        # miRNA batch has no offline augmentation.
        copy_one(html, "miRNA", cancer, group_label, norm, "none", param)


def main() -> None:
    migrate_rna()
    migrate_mirna()
    print(f"Copied {copied} learning_curve.html files into {DEST_ROOT.relative_to(ROOT)}/")
    if skipped:
        print(f"Skipped {len(skipped)} unexpected paths:")
        for s in skipped:
            print("  ", s)


if __name__ == "__main__":
    main()
