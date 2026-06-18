"""
sample_size_learning_curves.py
==============================
Sample-size calculation + learning curves for 24 cancers, mirroring
``Synthesize_Example.ipynb`` (SyNG-BTS / SyntheSize methodology).

For every combination of
    cancer (24) x normalisation (raw, TC, DESeq) x model (CVAE1-5/-10/-20)
    = 24 x 3 x 3 = 216 combinations
it evaluates classifier learning curves on:
    * REAL data      -> processed/<dataset><suffix>.csv
    * GENERATED data -> synthetic_output/<dataset>/<norm>/<model>/generated.csv

Shared plotting/evaluation logic lives in ``learning_curve_utils.py``.
Outputs follow the merge-friendly layout documented there (cohort
``microrna_24``).

Usage
-----
    python sample_size_learning_curves.py                  # full run (216 combos)
    python sample_size_learning_curves.py --quick          # tiny smoke test
    python sample_size_learning_curves.py --cancers BLCA_histologic_subtype
    python sample_size_learning_curves.py --norms raw --models CVAE1-10
    python sample_size_learning_curves.py --force          # recompute everything
    python sample_size_learning_curves.py --plotly-offline # embed plotly.js in each HTML
"""

from __future__ import annotations

import argparse
import os
import sys
import traceback
import warnings

warnings.filterwarnings(
    "ignore",
    message="l1_ratios parameter is only used when penalty is 'elasticnet'.*",
    category=UserWarning,
)

import numpy as np
import pandas as pd

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

from learning_curve_utils import (  # noqa: E402
    METHODS,
    combo_done,
    evaluate_metrics,
    load_features_groups,
    make_combo_id,
    make_sample_sizes,
    real_cache_matches,
    run_learning_curve_combo,
    write_index,
    write_manifest,
    write_real_cache_meta,
)

BASE_DIR = _SCRIPT_DIR
PROCESSED_DIR = os.path.join(BASE_DIR, "processed")
SYNTH_DIR = os.path.join(BASE_DIR, "synthetic_output")
OUT_ROOT = os.path.join(BASE_DIR, "learning_curve_output")
COHORT = "microrna_24"
COHORT_OUT = os.path.join(OUT_ROOT, "cohorts", COHORT)

NORMS = [
    ("raw", "_filtered.csv"),
    ("TC", "_filtered_TC.csv"),
    ("DESeq", "_filtered_DESeq.csv"),
]
MODELS = ["CVAE1-5", "CVAE1-10", "CVAE1-20"]

N_DRAWS = 5
N_SIZES = 12
APPLY_LOG = False
STATIC_METRIC = "f1_score"
RANDOM_SEED = 42


def dataset_names() -> list[str]:
    names = set()
    for f in os.listdir(PROCESSED_DIR):
        if f.endswith("_filtered.csv") and "_TC" not in f and "_DESeq" not in f:
            names.add(f.replace("_filtered.csv", ""))
    return sorted(names)


def main():
    global OUT_ROOT, COHORT_OUT

    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--cancers", nargs="*", default=None)
    ap.add_argument("--norms", nargs="*", default=None)
    ap.add_argument("--models", nargs="*", default=None)
    ap.add_argument("--methods", nargs="*", default=None)
    ap.add_argument("--n-draws", type=int, default=N_DRAWS)
    ap.add_argument("--n-sizes", type=int, default=N_SIZES)
    ap.add_argument("--metric", default="f1_score")
    ap.add_argument("--out-dir", default=OUT_ROOT,
                    help="Top-level output root (default: learning_curve_output/).")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--plotly-offline", action="store_true")
    args = ap.parse_args()
    np.random.seed(RANDOM_SEED)

    cancers = args.cancers or dataset_names()
    norms = [(lbl, suf) for lbl, suf in NORMS
             if args.norms is None or lbl in args.norms]
    models = [m for m in MODELS if args.models is None or m in args.models]
    methods = args.methods or list(METHODS)
    n_draws = args.n_draws
    n_sizes = args.n_sizes
    static_metric = args.metric

    OUT_ROOT = args.out_dir
    COHORT_OUT = os.path.join(OUT_ROOT, "cohorts", COHORT)
    os.makedirs(COHORT_OUT, exist_ok=True)

    if args.quick:
        cancers = cancers[:1]
        norms = [n for n in norms if n[0] == "raw"] or norms[:1]
        models = ["CVAE1-10"] if "CVAE1-10" in models else models[:1]
        methods = ["LOGIS", "RF"]
        n_draws, n_sizes = 2, 4

    total = len(cancers) * len(norms) * len(models)
    print(f"Cohort       : {COHORT}")
    print(f"Cancers      : {len(cancers)}")
    print(f"Norms        : {[n for n, _ in norms]}")
    print(f"Models       : {models}")
    print(f"Output       : {COHORT_OUT}")
    print(f"Total combos : {total}\n")

    summary_rows, manifest_rows, all_points = [], [], []
    combo_idx = 0

    for cancer in cancers:
        for norm_label, norm_suffix in norms:
            real_path = os.path.join(PROCESSED_DIR, f"{cancer}{norm_suffix}")
            norm_dir = os.path.join(COHORT_OUT, cancer, norm_label)
            os.makedirs(norm_dir, exist_ok=True)

            metrics_real = None
            real_csv = os.path.join(norm_dir, "metrics_real.csv")
            real_status = "pending"
            cache_ok = real_cache_matches(
                real_csv,
                methods=methods,
                n_draws=n_draws,
                n_sizes=n_sizes,
                apply_log=APPLY_LOG,
            )
            if cache_ok and not args.force:
                metrics_real = pd.read_csv(real_csv)
                real_status = "cached"
            elif os.path.exists(real_path):
                data_r, groups_r = load_features_groups(real_path)
                if data_r is None:
                    real_status = "real_no_groups"
                else:
                    sizes_r = make_sample_sizes(groups_r, n_sizes)
                    if not sizes_r:
                        real_status = "real_no_feasible_sizes"
                    else:
                        try:
                            metrics_real = evaluate_metrics(
                                data_r, groups_r, sizes_r,
                                n_draws, methods, APPLY_LOG,
                            )
                            metrics_real.to_csv(real_csv, index=False)
                            write_real_cache_meta(
                                real_csv,
                                methods=methods,
                                n_draws=n_draws,
                                n_sizes=n_sizes,
                                apply_log=APPLY_LOG,
                            )
                            real_status = "ok"
                        except Exception as exc:
                            real_status = f"real_error: {exc}"
                            traceback.print_exc()
            else:
                real_status = "real_file_missing"

            for model in models:
                combo_idx += 1
                combo_id = make_combo_id(COHORT, cancer, norm_label, model)
                title = f"{cancer} | {norm_label} | {model}"
                print(f"\n[{combo_idx}/{total}] {title}")

                model_dir = os.path.join(norm_dir, model)
                gen_path = os.path.join(SYNTH_DIR, cancer, norm_label, model, "generated.csv")

                if combo_done(model_dir, static_metric) and not args.force:
                    print("    already done — skipping")
                    status, secs = "skipped_done", 0.0
                    p = os.path.join(model_dir, "points.csv")
                    if os.path.exists(p):
                        all_points.append(pd.read_csv(p))
                else:
                    data_g, groups_g = (None, None)
                    if os.path.exists(gen_path):
                        data_g, groups_g = load_features_groups(gen_path)

                    meta = dict(
                        combo_id=combo_id,
                        cohort=COHORT,
                        cancer=cancer,
                        cancer_prefix=cancer,
                        norm=norm_label,
                        model=model,
                        off_aug="",
                        batch=1,
                        data_type="miRNA",
                        model_type="CVAE",
                        title=title,
                    )
                    status, points, secs = run_learning_curve_combo(
                        model_dir,
                        metrics_real,
                        data_g,
                        groups_g,
                        meta,
                        n_draws=n_draws,
                        n_sizes=n_sizes,
                        apply_log=APPLY_LOG,
                        methods=methods,
                        static_metric=static_metric,
                        plotly_offline=args.plotly_offline,
                        force=args.force,
                    )
                    if points is not None and not points.empty:
                        all_points.append(points)
                    print(f"    -> {status}  ({secs}s)")

                html_rel = os.path.relpath(
                    os.path.join(model_dir, "learning_curve.html"), OUT_ROOT,
                )
                png_rel = os.path.relpath(
                    os.path.join(model_dir, f"learning_curve_{static_metric}.png"),
                    OUT_ROOT,
                )
                points_rel = os.path.relpath(
                    os.path.join(model_dir, "points.csv"), OUT_ROOT,
                )
                summary_rows.append(dict(
                    combo_id=combo_id,
                    cohort=COHORT,
                    cancer=cancer,
                    norm=norm_label,
                    model=model,
                    off_aug="",
                    status=status,
                    real_status=real_status,
                    seconds=secs,
                ))
                manifest_rows.append(dict(
                    combo_id=combo_id,
                    cohort=COHORT,
                    cancer=cancer,
                    cancer_prefix=cancer,
                    norm=norm_label,
                    model=model,
                    off_aug="",
                    batch=1,
                    data_type="miRNA",
                    model_type="CVAE",
                    status=status,
                    title=title,
                    html_relpath=html_rel,
                    png_relpath=png_rel,
                    points_relpath=points_rel,
                    seconds=secs,
                ))

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(os.path.join(COHORT_OUT, "summary.csv"), index=False)
    if all_points:
        pd.concat(all_points, ignore_index=True).to_csv(
            os.path.join(COHORT_OUT, "all_points_long.csv"), index=False,
        )

    write_manifest(OUT_ROOT, manifest_rows, COHORT_OUT)
    write_index(COHORT_OUT, summary_df, static_metric=static_metric)

    ok = summary_df["status"].str.startswith("ok").sum()
    print(f"\n{'='*60}")
    print(f"Done. {ok}/{len(summary_df)} combos produced figures.")
    print(f"Merge manifest -> {os.path.join(OUT_ROOT, 'manifest.csv')}")


if __name__ == "__main__":
    main()
