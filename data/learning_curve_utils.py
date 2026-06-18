"""Shared helpers for sample-size learning-curve pipelines."""

from __future__ import annotations

import json
import os
import time
import traceback
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import approx_fprime, curve_fit
from scipy.stats import norm

from syng_bts import evaluate_sample_sizes, plot_sample_sizes

# ── Defaults (callers may override module-level METHODS / STATIC_METRIC) ─────
METHODS = ["LOGIS", "SVM", "KNN", "RF", "XGB"]
ALL_METRICS = ["f1_score", "accuracy", "auc"]
STATIC_METRIC = "f1_score"
N_SPLITS = 5
PLOT_YLIM = (0.4, 1.0)
# LogisticRegressionCV uses cv=5; outer 5-fold train ~80% of n must leave
# enough per-class samples for the inner CV to succeed.
MIN_N_FOR_CLASSIFIERS = 20

MANIFEST_COLUMNS = [
    "combo_id",
    "cohort",
    "cancer",
    "cancer_prefix",
    "norm",
    "model",
    "off_aug",
    "batch",
    "data_type",
    "model_type",
    "status",
    "title",
    "html_relpath",
    "png_relpath",
    "points_relpath",
    "seconds",
]


def make_combo_id(
    cohort: str,
    cancer: str,
    norm: str,
    model: str,
    off_aug: str = "",
    batch: int = 1,
) -> str:
    """Stable identifier for merging results across cohorts."""
    parts = [cohort, cancer, norm]
    if off_aug:
        parts.append(off_aug)
    parts.extend([model, f"batch{batch}"])
    return ":".join(parts)


def normalize_groups(raw_groups: pd.Series) -> pd.Series | None:
    """Return a clean 2-class string label series, or None if unusable."""
    unique_vals = raw_groups.dropna().unique()
    if len(unique_vals) != 2:
        return None
    if pd.api.types.is_numeric_dtype(raw_groups):
        return raw_groups.apply(
            lambda x: str(int(round(x))) if pd.notna(x) else "NA"
        )
    return raw_groups.astype(str)


def load_features_groups(filepath: str):
    """Load CSV → (feature DataFrame, groups Series) or (None, None)."""
    df = pd.read_csv(filepath)
    if "groups" not in df.columns:
        return None, None
    groups = normalize_groups(df["groups"])
    if groups is None:
        return None, None
    drop_cols = [
        c for c in df.columns
        if c == "groups" or not pd.api.types.is_numeric_dtype(df[c])
    ]
    data = df.drop(columns=drop_cols)
    return data.reset_index(drop=True), groups.reset_index(drop=True)


def load_real_test_csv(filepath: str):
    """Load augmentation test CSV (drops optional ``samples`` column)."""
    df = pd.read_csv(filepath)
    if "samples" in df.columns:
        df = df.drop(columns=["samples"])
    if "groups" not in df.columns:
        return None, None
    groups = normalize_groups(df["groups"])
    if groups is None:
        return None, None
    drop_cols = [
        c for c in df.columns
        if c == "groups" or not pd.api.types.is_numeric_dtype(df[c])
    ]
    data = df.drop(columns=drop_cols)
    return data.reset_index(drop=True), groups.reset_index(drop=True)


def load_generated_with_groups(
    gen_path: str,
    groups_path: str,
    expr_cols: list[str] | None = None,
):
    """Load generated expression matrix + sidecar group labels."""
    if not os.path.exists(gen_path) or not os.path.exists(groups_path):
        return None, None

    gen = pd.read_csv(gen_path)
    groups_raw = pd.read_csv(groups_path).iloc[:, 0]
    if len(groups_raw) != len(gen):
        return None, None

    if expr_cols is not None:
        if len(gen.columns) != len(expr_cols):
            return None, None
        gen.columns = expr_cols

    groups = normalize_groups(groups_raw)
    if groups is None:
        return None, None

    non_numeric = [c for c in gen.columns if not pd.api.types.is_numeric_dtype(gen[c])]
    if non_numeric:
        gen = gen.drop(columns=non_numeric)
    return gen.reset_index(drop=True), groups.reset_index(drop=True)


def _allocate_stratified_counts(total_size: int, group_counts: dict[str, int]) -> dict[str, int]:
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
    group_counts = {str(g): int(c) for g, c in groups.value_counts().items()}
    total = sum(group_counts.values())
    candidates = sorted({int(round(x)) for x in np.linspace(N_SPLITS * 2, total, 400)})
    feasible = [
        s for s in candidates
        if s >= MIN_N_FOR_CLASSIFIERS and _feasible(s, group_counts)
    ]
    if not feasible:
        return []
    if feasible[-1] != total and _feasible(total, group_counts):
        feasible.append(total)
        feasible = sorted(set(feasible))
    idx = np.linspace(0, len(feasible) - 1, min(n_sizes, len(feasible)))
    return sorted({feasible[int(round(i))] for i in idx})


def _power_law(x, a, b, c):
    return (1 - a) - b * (x ** c)


def fit_iplf(ns: np.ndarray, ys: np.ndarray):
    try:
        popt, pcov = curve_fit(_power_law, ns, ys, p0=[0, 1, -0.5], maxfev=50000)
        return popt, pcov, True
    except (RuntimeError, ValueError):
        return None, None, False


def predict_with_ci(popt, pcov, xs: np.ndarray):
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
    sub = metrics[metrics["method"] == method]
    agg = (
        sub.groupby("total_size")
        .agg(
            observed_mean=(metric, "mean"),
            observed_std=(metric, "std"),
            n_draws=(metric, "size"),
        )
        .reset_index()
        .rename(columns={"total_size": "n"})
        .sort_values("n")
    )
    agg["observed_std"] = agg["observed_std"].fillna(0.0)
    return agg


def build_points_and_fits(
    metrics_real,
    metrics_gen,
    extra_meta: dict[str, Any] | None = None,
):
    """Return (points_df, fits_df) with optional metadata columns on every row."""
    extra_meta = extra_meta or {}
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

                for _, r in mt.iterrows():
                    point_rows.append(dict(
                        source=source,
                        method=method,
                        metric=metric,
                        kind="observed",
                        n=int(r["n"]),
                        value=float(r["observed_mean"]),
                        observed_std=float(r["observed_std"]),
                        ci_low=np.nan,
                        ci_high=np.nan,
                        n_draws=int(r["n_draws"]),
                        **extra_meta,
                    ))

                popt, pcov, ok = fit_iplf(ns, ys) if len(ns) >= 3 else (None, None, False)
                fit_rows.append(dict(
                    source=source,
                    method=method,
                    metric=metric,
                    fit_ok=ok,
                    a=(float(popt[0]) if ok else np.nan),
                    b=(float(popt[1]) if ok else np.nan),
                    c=(float(popt[2]) if ok else np.nan),
                    **extra_meta,
                ))
                if ok:
                    grid = np.linspace(ns.min(), ns.max(), 100)
                    pred, lo, hi = predict_with_ci(popt, pcov, grid)
                    for xg, pg, lg, hg in zip(grid, pred, lo, hi):
                        point_rows.append(dict(
                            source=source,
                            method=method,
                            metric=metric,
                            kind="fitted",
                            n=float(xg),
                            value=float(pg),
                            observed_std=np.nan,
                            ci_low=float(lg),
                            ci_high=float(hg),
                            n_draws=np.nan,
                            **extra_meta,
                        ))
    return pd.DataFrame(point_rows), pd.DataFrame(fit_rows)


def build_interactive_html(
    points: pd.DataFrame,
    html_path: str,
    title: str,
    plotly_offline: bool,
    static_metric: str | None = None,
):
    static_metric = static_metric or STATIC_METRIC
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
    present_methods = [m for m in METHODS if not points[points["method"] == m].empty]
    if not present_methods:
        return False
    n_rows = len(present_methods)

    fig = make_subplots(
        rows=n_rows, cols=2,
        subplot_titles=[f"{m}: {s.capitalize()}"
                        for m in present_methods for s in sources],
        horizontal_spacing=0.08,
        vertical_spacing=0.06 if n_rows > 1 else 0.1,
    )

    colors = {"real": "#d62728", "generated": "#1f77b4"}
    trace_metric = []

    def add(metric, visible):
        for ri, method in enumerate(present_methods, start=1):
            for ci, source in enumerate(sources, start=1):
                sel = points[
                    (points["method"] == method)
                    & (points["source"] == source)
                    & (points["metric"] == metric)
                ]
                obs = sel[sel["kind"] == "observed"].sort_values("n")
                fit = sel[sel["kind"] == "fitted"].sort_values("n")
                col = colors[source]

                if not fit.empty:
                    fig.add_trace(go.Scatter(
                        x=list(fit["n"]) + list(fit["n"][::-1]),
                        y=list(fit["ci_high"]) + list(fit["ci_low"][::-1]),
                        fill="toself", fillcolor=col, opacity=0.18,
                        line=dict(width=0), hoverinfo="skip",
                        showlegend=False, visible=visible,
                    ), row=ri, col=ci)
                    trace_metric.append(metric)
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
                        error_y=dict(
                            type="data", array=obs["observed_std"],
                            visible=True, color=col, thickness=1,
                        ),
                        name="Observed", showlegend=False, visible=visible,
                        customdata=obs["observed_std"],
                        hovertemplate=(
                            "n=%{x:.0f}<br>mean=%{y:.3f}"
                            "<br>sd=%{customdata:.3f}<extra></extra>"
                        ),
                    ), row=ri, col=ci)
                    trace_metric.append(metric)

    for metric in ALL_METRICS:
        add(metric, visible=(metric == static_metric))

    for r in range(1, n_rows + 1):
        for c in (1, 2):
            fig.update_yaxes(range=list(PLOT_YLIM), row=r, col=c)
            fig.update_xaxes(title_text="Sample size", row=r, col=c)

    buttons = []
    for metric in ALL_METRICS:
        vis = [tm == metric for tm in trace_metric]
        buttons.append(dict(
            label=metric, method="update",
            args=[{"visible": vis}, {"title": f"{title} — {metric}"}],
        ))
    fig.update_layout(
        title=f"{title} — {static_metric}",
        updatemenus=[dict(
            active=ALL_METRICS.index(static_metric),
            buttons=buttons, x=1.0, xanchor="right",
            y=1.06, yanchor="bottom", direction="down",
        )],
        height=360 * n_rows + 80,
        width=1100,
        margin=dict(t=90, l=60, r=30, b=50),
        template="plotly_white",
    )
    fig.write_html(
        html_path,
        include_plotlyjs=(True if plotly_offline else "cdn"),
        full_html=True,
    )
    return True


def save_static_figure(metrics_real, metrics_gen, png_path, n_target, metric):
    fig = plot_sample_sizes(
        metric_real=metrics_real,
        n_target=n_target,
        metric_generated=metrics_gen,
        metric_name=metric,
    )
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def combo_done(combo_dir: str, static_metric: str | None = None) -> bool:
    static_metric = static_metric or STATIC_METRIC
    return all(os.path.exists(os.path.join(combo_dir, f)) for f in (
        "points.csv",
        f"learning_curve_{static_metric}.png",
        "learning_curve.html",
    ))


def real_cache_meta_path(real_csv: str) -> str:
    return real_csv.replace(".csv", ".meta.json")


def write_real_cache_meta(
    real_csv: str,
    *,
    methods: list[str],
    n_draws: int,
    n_sizes: int,
    apply_log: bool,
) -> None:
    meta = {
        "methods": sorted(methods),
        "n_draws": int(n_draws),
        "n_sizes": int(n_sizes),
        "apply_log": bool(apply_log),
    }
    with open(real_cache_meta_path(real_csv), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)


def real_cache_matches(
    real_csv: str,
    *,
    methods: list[str],
    n_draws: int,
    n_sizes: int,
    apply_log: bool,
) -> bool:
    """True when cached real metrics match the current evaluation settings."""
    if not os.path.exists(real_csv):
        return False
    meta_path = real_cache_meta_path(real_csv)
    if not os.path.exists(meta_path):
        return False
    with open(meta_path, encoding="utf-8") as f:
        meta = json.load(f)
    return (
        sorted(methods) == meta.get("methods")
        and int(n_draws) == int(meta.get("n_draws", -1))
        and int(n_sizes) == int(meta.get("n_sizes", -1))
        and bool(apply_log) == bool(meta.get("apply_log"))
    )


def evaluate_metrics(data, groups, sample_sizes, n_draws, methods, apply_log):
    return evaluate_sample_sizes(
        data=data,
        sample_sizes=sample_sizes,
        groups=groups,
        n_draws=n_draws,
        methods=methods,
        apply_log=apply_log,
        verbose="minimal",
    )


def run_learning_curve_combo(
    combo_dir: str,
    metrics_real,
    data_g,
    groups_g,
    meta: dict[str, Any],
    *,
    n_draws: int,
    n_sizes: int,
    apply_log: bool,
    methods: list[str],
    static_metric: str,
    plotly_offline: bool,
    force: bool,
) -> tuple[str, pd.DataFrame | None, float]:
    """Evaluate one (real, generated) pair and write figures under ``combo_dir``."""
    os.makedirs(combo_dir, exist_ok=True)
    title = meta.get("title", meta.get("combo_id", "learning curve"))

    if combo_done(combo_dir, static_metric) and not force:
        points_path = os.path.join(combo_dir, "points.csv")
        points = pd.read_csv(points_path) if os.path.exists(points_path) else None
        return "skipped_done", points, 0.0

    t0 = time.time()
    status = "ok"
    metrics_gen = None
    sizes_g: list[int] = []
    points = None

    try:
        if data_g is None or groups_g is None:
            status = "generated_file_missing"
        else:
            sizes_g = make_sample_sizes(groups_g, n_sizes)
            if not sizes_g:
                status = "generated_no_feasible_sizes"
            else:
                metrics_gen = evaluate_metrics(
                    data_g, groups_g, sizes_g, n_draws, methods, apply_log,
                )
                metrics_gen.to_csv(
                    os.path.join(combo_dir, "metrics_generated.csv"), index=False,
                )

        points, fits = build_points_and_fits(metrics_real, metrics_gen, extra_meta=meta)
        if not points.empty:
            points.to_csv(os.path.join(combo_dir, "points.csv"), index=False)
        if not fits.empty:
            fits.to_csv(os.path.join(combo_dir, "fits.csv"), index=False)

        n_target = int(sizes_g[-1]) if sizes_g else 1000
        png_path = os.path.join(combo_dir, f"learning_curve_{static_metric}.png")
        if metrics_real is not None and metrics_gen is not None:
            save_static_figure(
                metrics_real, metrics_gen, png_path, n_target=n_target, metric=static_metric,
            )
        elif metrics_real is not None or metrics_gen is not None:
            one = metrics_real if metrics_real is not None else metrics_gen
            save_static_figure(one, None, png_path, n_target=n_target, metric=static_metric)
            if status == "ok":
                status = "ok_one_sided"

        if not points.empty:
            build_interactive_html(
                points,
                os.path.join(combo_dir, "learning_curve.html"),
                title=title,
                plotly_offline=plotly_offline,
                static_metric=static_metric,
            )

        if metrics_real is None and metrics_gen is None and status == "ok":
            status = "no_data"

    except Exception as exc:
        status = f"error: {exc}"
        traceback.print_exc()
        points = None

    return status, points, round(time.time() - t0, 1)


def write_index(
    out_dir: str,
    summary_df: pd.DataFrame,
    static_metric: str | None = None,
    extra_columns: list[str] | None = None,
):
    static_metric = static_metric or STATIC_METRIC
    extra_columns = extra_columns or []
    base_cols = ["cancer", "norm", "model"] + extra_columns
    header = "".join(f"<th>{c}</th>" for c in base_cols)
    header += "<th>Status</th><th>Interactive</th><th>Static</th>"

    rows = []
    sort_cols = [c for c in base_cols if c in summary_df.columns]
    for _, r in summary_df.sort_values(sort_cols).iterrows():
        rel_parts = [str(r[c]) for c in base_cols if c in summary_df.columns]
        html_rel = os.path.join(*rel_parts, "learning_curve.html")
        png_rel = os.path.join(*rel_parts, f"learning_curve_{static_metric}.png")
        link = (
            f'<a href="{html_rel}">interactive</a>'
            if os.path.exists(os.path.join(out_dir, html_rel)) else "—"
        )
        png = (
            f'<a href="{png_rel}">png</a>'
            if os.path.exists(os.path.join(out_dir, png_rel)) else "—"
        )
        cells = "".join(f"<td>{r[c]}</td>" for c in base_cols if c in summary_df.columns)
        rows.append(
            f"<tr>{cells}<td>{r['status']}</td><td>{link}</td><td>{png}</td></tr>"
        )

    html = f"""<!doctype html><html><head><meta charset="utf-8">
<title>Learning-curve index</title>
<style>body{{font-family:sans-serif;margin:24px}}table{{border-collapse:collapse}}
td,th{{border:1px solid #ccc;padding:4px 10px;font-size:14px}}
th{{background:#f3f3f3}}tr:hover{{background:#fafafa}}</style></head><body>
<h2>Sample-size learning curves</h2>
<p>{len(summary_df)} combinations · static metric = {static_metric}</p>
<table><tr>{header}</tr>
{''.join(rows)}
</table></body></html>"""
    with open(os.path.join(out_dir, "index.html"), "w", encoding="utf-8") as f:
        f.write(html)


def write_manifest(out_root: str, manifest_rows: list[dict], cohort_out_dir: str):
    """Write cohort summary + append/update the top-level merge manifest."""
    if not manifest_rows:
        return

    manifest_df = pd.DataFrame(manifest_rows)
    for col in MANIFEST_COLUMNS:
        if col not in manifest_df.columns:
            manifest_df[col] = ""
    manifest_df = manifest_df[MANIFEST_COLUMNS]

    cohort_manifest = os.path.join(cohort_out_dir, "manifest.csv")
    manifest_df.to_csv(cohort_manifest, index=False)

    top_manifest = os.path.join(out_root, "manifest.csv")
    if os.path.exists(top_manifest):
        existing = pd.read_csv(top_manifest)
        existing = existing[~existing["combo_id"].isin(manifest_df["combo_id"])]
        merged = pd.concat([existing, manifest_df], ignore_index=True)
    else:
        merged = manifest_df
    merged = merged.sort_values(["cohort", "cancer", "norm", "off_aug", "model"])
    merged.to_csv(top_manifest, index=False)
