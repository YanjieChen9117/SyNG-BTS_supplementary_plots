"""
learning_curves_fivesubtypes_cvae.py
====================================
Learning curves for local FiveSubtypes RNA augmentation results (COAD + SKCM).

Runs 2 cancers x 3 norms x 3 models x 2 off_aug = 36 combinations using data
from ``syng_bts/data/case/Augmentation_FiveSubtypes-2026-04-15/``.

Output layout (merge-friendly)
------------------------------
learning_curve_output/
  manifest.csv                         ← union index for ALL cohorts (append on rerun)
  cohorts/
    fivesubtypes_rna_cvae/
      manifest.csv                     ← this run only
      summary.csv
      all_points_long.csv              ← long table for cross-figure replotting
      index.html
      COAD/raw/offaug_none/CVAE1-50/
        learning_curve.html
        learning_curve_f1_score.png
        points.csv
        metrics_generated.csv
        fits.csv
      COAD/raw/metrics_real.csv        ← shared per (cancer, norm)

Merge strategy
--------------
* Each cohort writes rows to the top-level ``manifest.csv`` with a stable
  ``combo_id`` (``cohort:cancer:norm:off_aug:model:batchN``).
* Cohorts without off_aug (24-cancer miRNA) use ``off_aug=""`` in manifest.
* Downstream merge: ``pd.read_csv("learning_curve_output/manifest.csv")`` and
  filter/join on ``cohort``, ``cancer``, ``norm``, ``model``, ``off_aug``.
* For custom dashboards, concatenate ``all_points_long.csv`` from each cohort
  (same schema via ``points.csv`` metadata columns).

Usage
-----
    python learning_curves_fivesubtypes_cvae.py
    python learning_curves_fivesubtypes_cvae.py --quick
    python learning_curves_fivesubtypes_cvae.py --cancers COAD --force
    python learning_curves_fivesubtypes_cvae.py --plotly-offline
"""

from __future__ import annotations

import argparse
import os
import sys
import time
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

_REPO_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from learning_curve_utils import (  # noqa: E402
    METHODS,
    combo_done,
    evaluate_metrics,
    load_generated_with_groups,
    load_real_test_csv,
    make_combo_id,
    make_sample_sizes,
    real_cache_matches,
    run_learning_curve_combo,
    write_index,
    write_manifest,
    write_real_cache_meta,
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.path.join(
    _REPO_ROOT,
    "syng_bts/data/case/Augmentation_FiveSubtypes-2026-04-15",
)
OUT_ROOT = os.path.join(BASE_DIR, "learning_curve_output")
COHORT = "fivesubtypes_rna_cvae"
COHORT_OUT = os.path.join(OUT_ROOT, "cohorts", COHORT)

CANCER_CONFIGS = [
    ("COAD_5-2_filtered", "COAD"),
    ("SKCM_5-2_filtered", "SKCM"),
]
NORMS = ["raw", "TC", "DESeq"]
MODELS = ["CVAE1-50", "CVAE1-100", "CVAE1-200"]
OFF_AUGS = ["offaug_none", "offaug_AE_head_2"]
BATCH = 1
DATA_TYPE = "miRNA"

N_DRAWS = 5
N_SIZES = 12
APPLY_LOG = True
STATIC_METRIC = "f1_score"
RANDOM_SEED = 42


def paths_for_combo(cancer_prefix: str, norm: str, off_aug: str, model: str):
    batch_dir = os.path.join(DATA_ROOT, cancer_prefix, norm, f"batch_{BATCH}")
    dataname = f"{cancer_prefix}_{norm}_batch_{BATCH}"
    real_path = os.path.join(batch_dir, f"{dataname}_test.csv")
    gen_dir = os.path.join(batch_dir, off_aug, model, DATA_TYPE)
    gen_path = os.path.join(gen_dir, f"{dataname}_train_{model}_generated.csv")
    groups_path = os.path.join(gen_dir, f"{dataname}_train_{model}_generated_groups.csv")
    return real_path, gen_path, groups_path


def combo_out_dir(cancer_short: str, norm: str, off_aug: str, model: str) -> str:
    return os.path.join(COHORT_OUT, cancer_short, norm, off_aug, model)


def real_metrics_dir(cancer_short: str, norm: str) -> str:
    return os.path.join(COHORT_OUT, cancer_short, norm)


def main():
    global OUT_ROOT, COHORT_OUT

    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--cancers", nargs="*", default=None,
                    help="Subset: COAD SKCM (default: both).")
    ap.add_argument("--norms", nargs="*", default=None)
    ap.add_argument("--models", nargs="*", default=None)
    ap.add_argument("--off-augs", nargs="*", default=None,
                    dest="off_augs")
    ap.add_argument("--methods", nargs="*", default=None)
    ap.add_argument("--n-draws", type=int, default=N_DRAWS)
    ap.add_argument("--n-sizes", type=int, default=N_SIZES)
    ap.add_argument("--metric", default=STATIC_METRIC)
    ap.add_argument("--out-root", default=OUT_ROOT)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--quick", action="store_true",
                    help="Smoke test: COAD raw offaug_none CVAE1-50, 2 draws, 4 sizes.")
    ap.add_argument("--plotly-offline", action="store_true")
    args = ap.parse_args()
    np.random.seed(RANDOM_SEED)

    cancers = [
        (prefix, short) for prefix, short in CANCER_CONFIGS
        if args.cancers is None or short in args.cancers
    ]
    norms = [n for n in NORMS if args.norms is None or n in args.norms]
    models = [m for m in MODELS if args.models is None or m in args.models]
    off_augs = [o for o in OFF_AUGS if args.off_augs is None or o in args.off_augs]
    methods = args.methods or list(METHODS)
    n_draws = args.n_draws
    n_sizes = args.n_sizes
    static_metric = args.metric

    OUT_ROOT = args.out_root
    COHORT_OUT = os.path.join(OUT_ROOT, "cohorts", COHORT)
    os.makedirs(COHORT_OUT, exist_ok=True)

    if args.quick:
        cancers = cancers[:1]
        norms = ["raw"]
        models = ["CVAE1-50"]
        off_augs = ["offaug_none"]
        methods = ["LOGIS", "RF"]
        n_draws, n_sizes = 2, 4

    total = len(cancers) * len(norms) * len(models) * len(off_augs)
    print(f"Cohort       : {COHORT}")
    print(f"Data root    : {DATA_ROOT}")
    print(f"Cancers      : {[s for _, s in cancers]}")
    print(f"Norms        : {norms}")
    print(f"Models       : {models}")
    print(f"Off_aug      : {off_augs}")
    print(f"Methods      : {methods}")
    print(f"n_draws/n_sizes: {n_draws}/{n_sizes}")
    print(f"apply_log    : {APPLY_LOG}")
    print(f"Output       : {COHORT_OUT}")
    print(f"Total combos : {total}\n")

    summary_rows: list[dict] = []
    manifest_rows: list[dict] = []
    all_points: list[pd.DataFrame] = []
    combo_idx = 0

    for cancer_prefix, cancer_short in cancers:
        for norm in norms:
            norm_dir = real_metrics_dir(cancer_short, norm)
            os.makedirs(norm_dir, exist_ok=True)

            real_path, _, _ = paths_for_combo(cancer_prefix, norm, off_augs[0], models[0])
            metrics_real = None
            real_status = "pending"
            real_csv = os.path.join(norm_dir, "metrics_real.csv")
            expr_cols: list[str] | None = None

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
                if os.path.exists(real_path):
                    data_probe, _ = load_real_test_csv(real_path)
                    if data_probe is not None:
                        expr_cols = list(data_probe.columns)
            elif os.path.exists(real_path):
                data_r, groups_r = load_real_test_csv(real_path)
                if data_r is None:
                    real_status = "real_no_groups"
                else:
                    expr_cols = list(data_r.columns)
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

            for off_aug in off_augs:
                for model in models:
                    combo_idx += 1
                    combo_id = make_combo_id(
                        COHORT, cancer_short, norm, model, off_aug, BATCH,
                    )
                    title = f"{cancer_short} | {norm} | {off_aug} | {model}"
                    print(f"\n[{combo_idx}/{total}] {title}")

                    out_dir = combo_out_dir(cancer_short, norm, off_aug, model)
                    real_path, gen_path, groups_path = paths_for_combo(
                        cancer_prefix, norm, off_aug, model,
                    )

                    if combo_done(out_dir, static_metric) and not args.force:
                        print("    already done — skipping")
                        status = "skipped_done"
                        secs = 0.0
                        points_path = os.path.join(out_dir, "points.csv")
                        if os.path.exists(points_path):
                            all_points.append(pd.read_csv(points_path))
                    else:
                        data_g, groups_g = load_generated_with_groups(
                            gen_path, groups_path, expr_cols=expr_cols,
                        )
                        meta = dict(
                            combo_id=combo_id,
                            cohort=COHORT,
                            cancer=cancer_short,
                            cancer_prefix=cancer_prefix,
                            norm=norm,
                            model=model,
                            off_aug=off_aug,
                            batch=BATCH,
                            data_type=DATA_TYPE,
                            model_type="CVAE",
                            title=title,
                        )
                        status, points, secs = run_learning_curve_combo(
                            out_dir,
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
                        os.path.join(out_dir, "learning_curve.html"), OUT_ROOT,
                    )
                    png_rel = os.path.relpath(
                        os.path.join(out_dir, f"learning_curve_{static_metric}.png"),
                        OUT_ROOT,
                    )
                    points_rel = os.path.relpath(
                        os.path.join(out_dir, "points.csv"), OUT_ROOT,
                    )

                    summary_rows.append(dict(
                        combo_id=combo_id,
                        cohort=COHORT,
                        cancer=cancer_short,
                        cancer_prefix=cancer_prefix,
                        norm=norm,
                        model=model,
                        off_aug=off_aug,
                        batch=BATCH,
                        data_type=DATA_TYPE,
                        model_type="CVAE",
                        status=status,
                        real_status=real_status,
                        title=title,
                        seconds=secs,
                    ))
                    manifest_rows.append(dict(
                        combo_id=combo_id,
                        cohort=COHORT,
                        cancer=cancer_short,
                        cancer_prefix=cancer_prefix,
                        norm=norm,
                        model=model,
                        off_aug=off_aug,
                        batch=BATCH,
                        data_type=DATA_TYPE,
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
    write_index(
        COHORT_OUT,
        summary_df,
        static_metric=static_metric,
        extra_columns=["off_aug"],
    )

    ok = summary_df["status"].str.startswith("ok").sum()
    print(f"\n{'='*60}")
    print(f"Done. {ok}/{len(summary_df)} combos produced figures.")
    print(f"Cohort summary -> {os.path.join(COHORT_OUT, 'summary.csv')}")
    print(f"Cohort index   -> {os.path.join(COHORT_OUT, 'index.html')}")
    print(f"Merge manifest -> {os.path.join(OUT_ROOT, 'manifest.csv')}")


if __name__ == "__main__":
    main()
