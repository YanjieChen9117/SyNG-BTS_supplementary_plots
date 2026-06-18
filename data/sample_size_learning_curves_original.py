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
using ``evaluate_sample_sizes`` (all 5 classifiers) and renders the same
"rows = classifier, columns = real | generated" figure as the example
notebook via ``plot_sample_sizes``.

Per combination it writes
--------------------------
out/<cancer>/<norm>/
    metrics_real.csv                 <- per-draw raw metrics on real data
                                        (shared by all 3 models, computed once)
    <model>/
        metrics_generated.csv        <- per-draw raw metrics on generated data
        points.csv                   <- tidy observed + fitted + 95% CI
                                        (everything needed for custom interactive plots)
        fits.csv                     <- IPLF fit params (a, b, c) per source/method/metric
        learning_curve_f1_score.png  <- STATIC figure (plot_sample_sizes)
        learning_curve.html          <- INTERACTIVE Plotly figure (metric dropdown)

Top-level outputs
-----------------
out/summary.csv          <- one row per combination with status + timings
out/all_points_long.csv  <- every point of every combo concatenated (cross-cancer interactive)
out/index.html           <- clickable index linking every interactive figure

Notes
-----
* The processed data is ALREADY log2(x + 1), so ``apply_log=False`` is used
  for both real and generated data (matching generate_synthetic.py).
* ``evaluate_sample_sizes`` always returns f1_score, accuracy AND auc in one
  pass, so the interactive HTML can offer all three via a dropdown even though
  the static PNG defaults to f1_score.
* Real metrics depend only on (cancer, norm) -- not on the model -- so they are
  computed once per (cancer, norm) and reused across the 3 models.
* The run is resumable: combos whose outputs already exist are skipped unless
  --force is given.

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
import json
import os
import time
import traceback
import warnings

# Harmless sklearn noise: LogisticRegressionCV warns that `l1_ratios` is unused
# under penalty='l2' (it is, by design — l1_ratios=0 == pure ridge). Silence it
# so the evaluation progress bar stays readable. Other warnings are kept.
warnings.filterwarnings(
    "ignore",
    message="l1_ratios parameter is only used when penalty is 'elasticnet'.*",
    category=UserWarning,
)

import matplotlib
matplotlib.use("Agg")            # non-interactive backend; must precede pyplot
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import approx_fprime, curve_fit
from scipy.stats import norm

from syng_bts import evaluate_sample_sizes, plot_sample_sizes

# ── Configuration ────────────────────────────────────────────────────────────
BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
PROCESSED_DIR = os.path.join(BASE_DIR, "processed")
SYNTH_DIR     = os.path.join(BASE_DIR, "synthetic_output")
DEFAULT_OUT   = os.path.join(BASE_DIR, "learning_curve_output")

NORMS = [
    ("raw",   "_filtered.csv"),
    ("TC",    "_filtered_TC.csv"),
    ("DESeq", "_filtered_DESeq.csv"),
]
MODELS  = ["CVAE1-5", "CVAE1-10", "CVAE1-20"]
METHODS = ["LOGIS", "SVM", "KNN", "RF", "XGB"]   # rows of every figure

N_DRAWS      = 5
N_SIZES      = 12          # ~12 sample sizes per panel (adaptive per dataset)
APPLY_LOG    = False       # processed data is already log2(x + 1)
STATIC_METRIC = "f1_score"
ALL_METRICS   = ["f1_score", "accuracy", "auc"]
RANDOM_SEED   = 42
N_SPLITS      = 5          # StratifiedKFold inside evaluate_sample_sizes

PLOT_YLIM = (0.4, 1.0)     # matches plot_sample_sizes


# ── Data loading ─────────────────────────────────────────────────────────────
def load_features_groups(filepath: str):
    """Load a CSV → (numeric feature DataFrame, clean string groups Series).

    Mirrors generate_synthetic.load_processed: drops the groups column and any
    non-numeric columns; normalises a numeric binary label to "0"/"1". Returns
    (None, None) when the file lacks a usable 2-class groups column.
    """
    df = pd.read_csv(filepath)

    if "groups" not in df.columns:
        return None, None

    raw_groups = df["groups"]
    unique_vals = raw_groups.dropna().unique()
    if len(unique_vals) != 2:
        return None, None

    if pd.api.types.is_numeric_dtype(raw_groups):
        groups = raw_groups.apply(lambda x: str(int(round(x))) if pd.notna(x) else "NA")
    else:
        groups = raw_groups.astype(str)

    drop_cols = [c for c in df.columns
                 if c == "groups" or not pd.api.types.is_numeric_dtype(df[c])]
    data = df.drop(columns=drop_cols)
    return data.reset_index(drop=True), groups.reset_index(drop=True)


def dataset_names() -> list[str]:
    """All dataset stems that have a raw _filtered.csv in processed/."""
    names = set()
    for f in os.listdir(PROCESSED_DIR):
        if f.endswith("_filtered.csv") and "_TC" not in f and "_DESeq" not in f:
            names.add(f.replace("_filtered.csv", ""))
    return sorted(names)


# ── Adaptive sample-size grid (respects evaluate_sample_sizes feasibility) ────
def _allocate_stratified_counts(total_size: int, group_counts: dict[str, int]) -> dict[str, int]:
    """Largest-remainder proportional allocation (copy of the package logic)."""
    total_available = sum(group_counts.values())
    groups = list(group_counts.keys())
    raw = {g: total_size * group_counts[g] / total_available for g in groups}
    alloc = {g: min(int(np.floor(raw[g])), group_counts[g]) for g in groups}
    remaining = total_size - sum(alloc.values())
    if remaining > 0:
        order = sorted(groups, key=lambda g: raw[g] - alloc[g], reverse=True)
        while remaining > 0:
            progressed = False
            for g in order:
                if alloc[g] < group_counts[g]:
                    alloc[g] += 1
                    remaining -= 1
                    progressed = True
                    if remaining == 0:
                        break
            if not progressed:
                break
    return alloc


def _feasible(s: int, group_counts: dict[str, int]) -> bool:
    n_classes = len(group_counts)
    if s < N_SPLITS * n_classes or s > sum(group_counts.values()):
        return False
    alloc = _allocate_stratified_counts(s, group_counts)
    return all(c >= N_SPLITS for c in alloc.values())


def make_sample_sizes(groups: pd.Series, n_sizes: int) -> list[int]:
    """Build up to n_sizes evenly-spaced, feasible sample sizes for stratified
    5-fold CV, with the maximum equal to the total number of available rows."""
    group_counts = {str(g): int(c) for g, c in groups.value_counts().items()}
    total = sum(group_counts.values())
    candidates = sorted({int(round(x)) for x in np.linspace(N_SPLITS * 2, total, 400)})
    feasible = [s for s in candidates if _feasible(s, group_counts)]
    if not feasible:
        return []
    if feasible[-1] != total and _feasible(total, group_counts):
        feasible.append(total)
        feasible = sorted(set(feasible))
    idx = np.linspace(0, len(feasible) - 1, min(n_sizes, len(feasible)))
    chosen = sorted({feasible[int(round(i))] for i in idx})
    return chosen


# ── IPLF curve fit (identical math to syng_bts._fit_curve) ───────────────────
def _power_law(x, a, b, c):
    return (1 - a) - b * (x ** c)


def fit_iplf(ns: np.ndarray, ys: np.ndarray):
    """Fit (1-a) - b*x^c. Returns (popt, pcov, ok)."""
    try:
        popt, pcov = curve_fit(_power_law, ns, ys, p0=[0, 1, -0.5], maxfev=50000)
        return popt, pcov, True
    except (RuntimeError, ValueError):
        return None, None, False


def predict_with_ci(popt, pcov, xs: np.ndarray):
    """Predicted values + 95% delta-method CI on grid xs."""
    pred = _power_law(xs, *popt)
    eps = np.sqrt(np.finfo(float).eps)
    jac = np.empty((len(xs), len(popt)))
    for i, x in enumerate(xs):
        jac[i] = approx_fprime([x], lambda z: _power_law(z[0], *popt), eps)
    var = np.sum((jac @ pcov) * jac, axis=1)
    sd = np.sqrt(np.clip(var, 0, None))
    t = norm.ppf(0.975)
    return pred, pred - t * sd, pred + t * sd


def mean_table(metrics: pd.DataFrame, method: str, metric: str) -> pd.DataFrame:
    """Mean (over draws) of `metric` per sample size for one classifier."""
    sub = metrics[metrics["method"] == method]
    agg = (sub.groupby("total_size")
              .agg(observed_mean=(metric, "mean"),
                   observed_std=(metric, "std"),
                   n_draws=(metric, "size"))
              .reset_index()
              .rename(columns={"total_size": "n"})
              .sort_values("n"))
    agg["observed_std"] = agg["observed_std"].fillna(0.0)
    return agg


# ── Per-combo point/fit extraction (for interactive + CSV) ───────────────────
def build_points_and_fits(metrics_real, metrics_gen, cancer, norm, model):
    """Return (points_df, fits_df) covering both sources, all methods/metrics."""
    point_rows, fit_rows = [], []
    sources = [("real", metrics_real), ("generated", metrics_gen)]

    for source, metrics in sources:
        if metrics is None or metrics.empty:
            continue
        for method in METHODS:
            if method not in set(metrics["method"]):
                continue
            for metric in ALL_METRICS:
                mt = mean_table(metrics, method, metric)
                if mt.empty:
                    continue
                ns = mt["n"].to_numpy(dtype=float)
                ys = mt["observed_mean"].to_numpy(dtype=float)

                # observed points
                for _, r in mt.iterrows():
                    point_rows.append(dict(
                        cancer=cancer, norm=norm, model=model, source=source,
                        method=method, metric=metric, kind="observed",
                        n=int(r["n"]), value=float(r["observed_mean"]),
                        observed_std=float(r["observed_std"]),
                        ci_low=np.nan, ci_high=np.nan, n_draws=int(r["n_draws"]),
                    ))

                # fitted curve + CI on a dense grid
                popt, pcov, ok = fit_iplf(ns, ys) if len(ns) >= 3 else (None, None, False)
                fit_rows.append(dict(
                    cancer=cancer, norm=norm, model=model, source=source,
                    method=method, metric=metric, fit_ok=ok,
                    a=(float(popt[0]) if ok else np.nan),
                    b=(float(popt[1]) if ok else np.nan),
                    c=(float(popt[2]) if ok else np.nan),
                ))
                if ok:
                    grid = np.linspace(ns.min(), ns.max(), 100)
                    pred, lo, hi = predict_with_ci(popt, pcov, grid)
                    for xg, pg, lg, hg in zip(grid, pred, lo, hi):
                        point_rows.append(dict(
                            cancer=cancer, norm=norm, model=model, source=source,
                            method=method, metric=metric, kind="fitted",
                            n=float(xg), value=float(pg),
                            observed_std=np.nan, ci_low=float(lg), ci_high=float(hg),
                            n_draws=np.nan,
                        ))
    return pd.DataFrame(point_rows), pd.DataFrame(fit_rows)


# ── Interactive Plotly figure (rows = method, cols = real | generated) ───────
def build_interactive_html(points: pd.DataFrame, html_path: str, title: str,
                           plotly_offline: bool):
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError:
        if not getattr(build_interactive_html, "_warned", False):
            print("    [note] plotly not installed — skipping interactive HTML. "
                  "Install with: pip install plotly")
            build_interactive_html._warned = True
        return False

    sources = ["real", "generated"]
    present_methods = [m for m in METHODS
                       if not points[points["method"] == m].empty]
    if not present_methods:
        return False
    n_rows = len(present_methods)

    fig = make_subplots(
        rows=n_rows, cols=2,
        subplot_titles=[f"{m}: {s.capitalize()}"
                        for m in present_methods for s in sources],
        horizontal_spacing=0.08, vertical_spacing=0.06 if n_rows > 1 else 0.1,
    )

    colors = {"real": "#d62728", "generated": "#1f77b4"}
    trace_metric = []          # parallel list: metric each trace belongs to

    def add(metric, visible):
        for ri, method in enumerate(present_methods, start=1):
            for ci, source in enumerate(sources, start=1):
                sel = points[(points["method"] == method)
                             & (points["source"] == source)
                             & (points["metric"] == metric)]
                obs = sel[sel["kind"] == "observed"].sort_values("n")
                fit = sel[sel["kind"] == "fitted"].sort_values("n")
                col = colors[source]

                if not fit.empty:
                    # CI band
                    fig.add_trace(go.Scatter(
                        x=list(fit["n"]) + list(fit["n"][::-1]),
                        y=list(fit["ci_high"]) + list(fit["ci_low"][::-1]),
                        fill="toself", fillcolor=col, opacity=0.18,
                        line=dict(width=0), hoverinfo="skip",
                        showlegend=False, visible=visible,
                    ), row=ri, col=ci)
                    trace_metric.append(metric)
                    # fitted line
                    fig.add_trace(go.Scatter(
                        x=fit["n"], y=fit["value"], mode="lines",
                        line=dict(color=col, dash="dash"),
                        name="Fitted", showlegend=False, visible=visible,
                        hovertemplate="n=%{x:.0f}<br>fit=%{y:.3f}<extra></extra>",
                    ), row=ri, col=ci)
                    trace_metric.append(metric)

                if not obs.empty:
                    fig.add_trace(go.Scatter(
                        x=obs["n"], y=obs["value"], mode="markers",
                        marker=dict(color=col, size=7),
                        error_y=dict(type="data", array=obs["observed_std"],
                                     visible=True, color=col, thickness=1),
                        name="Observed", showlegend=False, visible=visible,
                        customdata=obs["observed_std"],
                        hovertemplate=("n=%{x:.0f}<br>mean=%{y:.3f}"
                                       "<br>sd=%{customdata:.3f}<extra></extra>"),
                    ), row=ri, col=ci)
                    trace_metric.append(metric)

    for metric in ALL_METRICS:
        add(metric, visible=(metric == STATIC_METRIC))

    # y-axis range + labels
    for r in range(1, n_rows + 1):
        for c in (1, 2):
            fig.update_yaxes(range=list(PLOT_YLIM), row=r, col=c)
            fig.update_xaxes(title_text="Sample size", row=r, col=c)

    # metric dropdown
    buttons = []
    for metric in ALL_METRICS:
        vis = [tm == metric for tm in trace_metric]
        buttons.append(dict(label=metric, method="update",
                            args=[{"visible": vis},
                                  {"title": f"{title} — {metric}"}]))
    fig.update_layout(
        title=f"{title} — {STATIC_METRIC}",
        updatemenus=[dict(active=ALL_METRICS.index(STATIC_METRIC),
                          buttons=buttons, x=1.0, xanchor="right",
                          y=1.06, yanchor="bottom", direction="down")],
        height=360 * n_rows + 80, width=1100,
        margin=dict(t=90, l=60, r=30, b=50),
        template="plotly_white",
    )
    fig.write_html(html_path,
                   include_plotlyjs=(True if plotly_offline else "cdn"),
                   full_html=True)
    return True


# ── Static figure via the package's plot_sample_sizes ────────────────────────
def save_static_figure(metrics_real, metrics_gen, png_path, n_target, metric):
    fig = plot_sample_sizes(
        metric_real=metrics_real,
        n_target=n_target,
        metric_generated=metrics_gen,
        metric_name=metric,
    )
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ── Combo output existence (for resumability) ────────────────────────────────
def combo_done(model_dir: str) -> bool:
    return all(os.path.exists(os.path.join(model_dir, f))
               for f in ("points.csv",
                         f"learning_curve_{STATIC_METRIC}.png",
                         "learning_curve.html"))


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cancers", nargs="*", default=None,
                    help="Dataset stems to run (default: all 24).")
    ap.add_argument("--norms", nargs="*", default=None,
                    help="Subset of: raw TC DESeq (default: all).")
    ap.add_argument("--models", nargs="*", default=None,
                    help="Subset of: CVAE1-5 CVAE1-10 CVAE1-20 (default: all).")
    ap.add_argument("--methods", nargs="*", default=None,
                    help="Classifier subset (default: LOGIS SVM KNN RF XGB).")
    ap.add_argument("--n-draws", type=int, default=N_DRAWS)
    ap.add_argument("--n-sizes", type=int, default=N_SIZES)
    ap.add_argument("--metric", default="f1_score",
                    help="Metric for the static PNG (f1_score/accuracy/auc).")
    ap.add_argument("--out-dir", default=DEFAULT_OUT)
    ap.add_argument("--force", action="store_true",
                    help="Recompute combos even if outputs already exist.")
    ap.add_argument("--quick", action="store_true",
                    help="Smoke test: 1 cancer, raw, CVAE1-10, LOGIS+RF, n_draws=2, 4 sizes.")
    ap.add_argument("--plotly-offline", action="store_true",
                    help="Embed plotly.js in every HTML (larger files, works offline).")
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

    if args.quick:
        cancers = cancers[:1]
        norms = [n for n in norms if n[0] == "raw"] or norms[:1]
        models = ["CVAE1-10"] if "CVAE1-10" in models else models[:1]
        methods = ["LOGIS", "RF"]
        n_draws, n_sizes = 2, 4

    # Functions below read these module-level names; set them for this run.
    globals()["METHODS"] = methods
    globals()["STATIC_METRIC"] = static_metric

    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    total = len(cancers) * len(norms) * len(models)
    print(f"Cancers      : {len(cancers)}")
    print(f"Norms        : {[n for n, _ in norms]}")
    print(f"Models       : {models}")
    print(f"Methods       : {methods}")
    print(f"n_draws / n_sizes : {n_draws} / {n_sizes}")
    print(f"Static metric : {static_metric}")
    print(f"Output dir    : {out_dir}")
    print(f"Total combos  : {total}\n")

    summary_rows, all_points = [], []
    combo_idx = 0

    for cancer in cancers:
        for norm_label, norm_suffix in norms:
            real_path = os.path.join(PROCESSED_DIR, f"{cancer}{norm_suffix}")
            norm_dir = os.path.join(out_dir, cancer, norm_label)
            os.makedirs(norm_dir, exist_ok=True)

            # ── REAL metrics: compute once per (cancer, norm), then reuse ──
            metrics_real = None
            real_csv = os.path.join(norm_dir, "metrics_real.csv")
            real_status = "pending"
            if os.path.exists(real_csv) and not args.force:
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
                            metrics_real = evaluate_sample_sizes(
                                data=data_r, sample_sizes=sizes_r, groups=groups_r,
                                n_draws=n_draws, methods=methods, apply_log=APPLY_LOG,
                                verbose="minimal",
                            )
                            metrics_real.to_csv(real_csv, index=False)
                            real_status = "ok"
                        except Exception as exc:
                            real_status = f"real_error: {exc}"
                            traceback.print_exc()
            else:
                real_status = "real_file_missing"

            # ── GENERATED metrics: per model ──
            for model in models:
                combo_idx += 1
                tag = f"{cancer} | {norm_label} | {model}"
                print(f"\n[{combo_idx}/{total}] {tag}")

                model_dir = os.path.join(norm_dir, model)
                os.makedirs(model_dir, exist_ok=True)

                if combo_done(model_dir) and not args.force:
                    print("    already done — skipping (use --force to redo)")
                    summary_rows.append(dict(cancer=cancer, norm=norm_label,
                                             model=model, status="skipped_done",
                                             real_status=real_status, seconds=0))
                    p = os.path.join(model_dir, "points.csv")
                    if os.path.exists(p):
                        all_points.append(pd.read_csv(p))
                    continue

                gen_path = os.path.join(SYNTH_DIR, cancer, norm_label, model,
                                        "generated.csv")
                t0 = time.time()
                status = "ok"
                metrics_gen = None
                try:
                    if not os.path.exists(gen_path):
                        status = "generated_file_missing"
                    else:
                        data_g, groups_g = load_features_groups(gen_path)
                        if data_g is None:
                            status = "generated_no_groups"
                        else:
                            sizes_g = make_sample_sizes(groups_g, n_sizes)
                            if not sizes_g:
                                status = "generated_no_feasible_sizes"
                            else:
                                metrics_gen = evaluate_sample_sizes(
                                    data=data_g, sample_sizes=sizes_g,
                                    groups=groups_g, n_draws=n_draws,
                                    methods=methods, apply_log=APPLY_LOG,
                                    verbose="minimal",
                                )
                                metrics_gen.to_csv(
                                    os.path.join(model_dir, "metrics_generated.csv"),
                                    index=False)

                    # ── Points + fits (for interactive / custom plotting) ──
                    points, fits = build_points_and_fits(
                        metrics_real, metrics_gen, cancer, norm_label, model)
                    if not points.empty:
                        points.to_csv(os.path.join(model_dir, "points.csv"), index=False)
                        all_points.append(points)
                    if not fits.empty:
                        fits.to_csv(os.path.join(model_dir, "fits.csv"), index=False)

                    # ── Static figure (only when both sources available) ──
                    n_target = int(sizes_g[-1]) if metrics_gen is not None else 1000
                    if metrics_real is not None and metrics_gen is not None:
                        save_static_figure(
                            metrics_real, metrics_gen,
                            os.path.join(model_dir, f"learning_curve_{static_metric}.png"),
                            n_target=n_target, metric=static_metric)
                    elif metrics_real is not None or metrics_gen is not None:
                        # one-sided figure
                        one = metrics_real if metrics_real is not None else metrics_gen
                        save_static_figure(
                            one, None,
                            os.path.join(model_dir, f"learning_curve_{static_metric}.png"),
                            n_target=n_target, metric=static_metric)
                        if status == "ok":
                            status = "ok_one_sided"

                    # ── Interactive figure ──
                    if not points.empty:
                        build_interactive_html(
                            points,
                            os.path.join(model_dir, "learning_curve.html"),
                            title=tag, plotly_offline=args.plotly_offline)

                    if metrics_real is None and metrics_gen is None and status == "ok":
                        status = "no_data"

                except Exception as exc:
                    status = f"error: {exc}"
                    traceback.print_exc()

                secs = round(time.time() - t0, 1)
                print(f"    -> {status}  ({secs}s)")
                summary_rows.append(dict(cancer=cancer, norm=norm_label, model=model,
                                         status=status, real_status=real_status,
                                         seconds=secs))

    # ── Aggregate outputs ──
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(os.path.join(out_dir, "summary.csv"), index=False)

    if all_points:
        pd.concat(all_points, ignore_index=True).to_csv(
            os.path.join(out_dir, "all_points_long.csv"), index=False)

    write_index(out_dir, summary_df)

    ok = summary_df["status"].str.startswith("ok").sum()
    print(f"\n{'='*60}")
    print(f"Done. {ok}/{len(summary_df)} combos produced figures.")
    print(f"Summary  -> {os.path.join(out_dir, 'summary.csv')}")
    print(f"Index    -> {os.path.join(out_dir, 'index.html')}")


def write_index(out_dir: str, summary_df: pd.DataFrame):
    """Simple clickable index of every interactive figure."""
    rows = []
    for _, r in summary_df.sort_values(["cancer", "norm", "model"]).iterrows():
        html_rel = os.path.join(r["cancer"], r["norm"], r["model"],
                                "learning_curve.html")
        png_rel = os.path.join(r["cancer"], r["norm"], r["model"],
                               f"learning_curve_{STATIC_METRIC}.png")
        link = (f'<a href="{html_rel}">interactive</a>'
                if os.path.exists(os.path.join(out_dir, html_rel)) else "—")
        png = (f'<a href="{png_rel}">png</a>'
               if os.path.exists(os.path.join(out_dir, png_rel)) else "—")
        rows.append(f"<tr><td>{r['cancer']}</td><td>{r['norm']}</td>"
                    f"<td>{r['model']}</td><td>{r['status']}</td>"
                    f"<td>{link}</td><td>{png}</td></tr>")
    html = f"""<!doctype html><html><head><meta charset="utf-8">
<title>Learning-curve index</title>
<style>body{{font-family:sans-serif;margin:24px}}table{{border-collapse:collapse}}
td,th{{border:1px solid #ccc;padding:4px 10px;font-size:14px}}
th{{background:#f3f3f3}}tr:hover{{background:#fafafa}}</style></head><body>
<h2>Sample-size learning curves</h2>
<p>{len(summary_df)} combinations · static metric = {STATIC_METRIC}
(interactive figures include an f1/accuracy/auc dropdown).</p>
<table><tr><th>Cancer</th><th>Norm</th><th>Model</th><th>Status</th>
<th>Interactive</th><th>Static</th></tr>
{''.join(rows)}
</table></body></html>"""
    with open(os.path.join(out_dir, "index.html"), "w") as f:
        f.write(html)


if __name__ == "__main__":
    main()
