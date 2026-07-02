#!/usr/bin/env python3
"""Infer learning-curve view metadata from a Plotly ``learning_curve.html`` figure.

Structure is derived from the existing metric ``updatemenus`` visibility masks and
trace subplot assignments — no hard-coded trace counts.
"""
from __future__ import annotations

import re
from typing import Any

CLASSIFIERS = ["LOGIS", "SVM", "KNN", "RF", "XGB"]
DEFAULT_CLASSIFIER = "XGB"
DEFAULT_METRIC = "f1_score"
SINGLE_ROW_HEIGHT = 430
PLOT_WIDTH = 1100

REAL_DOMAIN = [0.0, 0.46]
GEN_DOMAIN = [0.54, 1.0]
FULL_X_DOMAIN = [0.0, 1.0]
FULL_Y_DOMAIN = [0.0, 1.0]
HIDDEN_DOMAIN = [0.0, 0.0]


def axis_ref_to_layout_key(ref: str) -> str:
    """Map Plotly trace axis refs (``x``, ``x9``) to layout keys (``xaxis``, ``xaxis9``)."""
    kind = ref[0]
    suffix = ref[1:]
    if not suffix:
        return f"{kind}axis"
    return f"{kind}axis{suffix}"


def parse_classifiers(annotations: list[dict[str, Any]]) -> list[str]:
    names: list[str] = []
    for ann in annotations:
        text = ann.get("text", "")
        if text.endswith(": Real"):
            names.append(text[: -len(": Real")])
    if names:
        return names
    return [ann.get("text", "").split(":")[0] for ann in annotations[::2]]


def equal_metric_blocks(trace_count: int, metric_count: int) -> list[list[int]]:
    if trace_count % metric_count != 0:
        raise ValueError(
            f"trace count {trace_count} is not divisible by {metric_count} metrics"
        )
    block_size = trace_count // metric_count
    return [
        list(range(m * block_size, (m + 1) * block_size))
        for m in range(metric_count)
    ]


def infer_view_spec(
    data: list[dict[str, Any]],
    layout: dict[str, Any],
    *,
    metrics: list[str] | None = None,
) -> dict[str, Any]:
    """Return a JSON-serialisable spec for the classifier / metric controller."""
    menus = layout.get("updatemenus") or []
    metric_blocks: list[list[int]]

    if menus:
        buttons = menus[0].get("buttons") or []
        if not buttons:
            raise ValueError("metric updatemenus has no buttons")
        parsed_metrics: list[str] = []
        metric_blocks = []
        for btn in buttons:
            args = btn.get("args") or []
            if not args:
                raise ValueError(f"metric button {btn.get('label')!r} has no args")
            vis = args[0].get("visible")
            if not isinstance(vis, list) or len(vis) != len(data):
                raise ValueError("visibility mask length does not match trace count")
            parsed_metrics.append(str(btn.get("label", "")))
            metric_blocks.append([i for i, show in enumerate(vis) if show])
        metrics = parsed_metrics
    else:
        if not metrics:
            metrics = ["f1_score", "accuracy", "auc"]
        metric_blocks = equal_metric_blocks(len(data), len(metrics))

    block_sizes = {len(block) for block in metric_blocks}
    if len(block_sizes) != 1:
        raise ValueError(f"metric blocks have inconsistent sizes: {block_sizes}")
    traces_per_metric = block_sizes.pop()

    annotations = layout.get("annotations") or []
    classifiers = parse_classifiers(annotations)
    if not classifiers:
        raise ValueError("no classifier annotations found")
    if classifiers != CLASSIFIERS:
        raise ValueError(f"unexpected classifier order: {classifiers}")

    subplot_groups: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    for idx in metric_blocks[0]:
        trace = data[idx]
        xref = trace.get("xaxis", "x")
        yref = trace.get("yaxis", "y")
        if (
            current is None
            or current["xref"] != xref
            or current["yref"] != yref
        ):
            current = {
                "offsets": [],
                "xref": xref,
                "yref": yref,
                "xkey": axis_ref_to_layout_key(xref),
                "ykey": axis_ref_to_layout_key(yref),
            }
            subplot_groups.append(current)
        current["offsets"].append(idx)

    n_subplots = len(subplot_groups)
    n_classifiers = len(classifiers)
    if n_subplots == n_classifiers:
        panels_per_classifier = 1
    elif n_subplots == 2 * n_classifiers:
        panels_per_classifier = 2
    else:
        raise ValueError(
            f"cannot map {n_subplots} subplots onto {n_classifiers} classifiers"
        )

    offset_sizes = {len(g["offsets"]) for g in subplot_groups}
    if len(offset_sizes) != 1:
        raise ValueError(f"inconsistent traces per subplot: {offset_sizes}")

    classifier_entries: list[dict[str, Any]] = []
    for ci, clf_name in enumerate(classifiers):
        panels: list[dict[str, Any]] = []
        for pi in range(panels_per_classifier):
            group = subplot_groups[ci * panels_per_classifier + pi]
            if panels_per_classifier == 1:
                ann_index = 2 * ci + 1
            else:
                ann_index = 2 * ci + pi
            panels.append(
                {
                    "xkey": group["xkey"],
                    "ykey": group["ykey"],
                    "offsets": group["offsets"],
                    "ann_index": ann_index,
                }
            )
        classifier_entries.append({"name": clf_name, "panels": panels})

    all_axis_keys = sorted(
        (k for k in layout if re.fullmatch(r"[xy]axis\d*", k)),
        key=lambda k: (k[0], int(re.sub(r"\D", "", k) or "0")),
    )

    original_height = int(layout.get("height", 1880))
    single_height = max(
        SINGLE_ROW_HEIGHT,
        int(round(original_height / n_classifiers)),
    )

    default_metric = DEFAULT_METRIC if DEFAULT_METRIC in metrics else metrics[0]
    default_classifier_idx = (
        classifiers.index(DEFAULT_CLASSIFIER)
        if DEFAULT_CLASSIFIER in classifiers
        else 0
    )

    return {
        "metrics": metrics,
        "classifiers": classifier_entries,
        "traces_per_metric": traces_per_metric,
        "trace_count": len(data),
        "panels_per_classifier": panels_per_classifier,
        "all_axis_keys": all_axis_keys,
        "default_metric": default_metric,
        "default_classifier": classifiers[default_classifier_idx],
        "single_height": single_height,
        "width": int(layout.get("width", PLOT_WIDTH)),
        "annotation_count": len(annotations),
    }


def classifier_index(spec: dict[str, Any], name: str) -> int:
    for i, entry in enumerate(spec["classifiers"]):
        if entry["name"] == name:
            return i
    raise ValueError(f"classifier {name!r} not in spec")


def metric_index(spec: dict[str, Any], name: str) -> int:
    for i, metric in enumerate(spec["metrics"]):
        if metric == name:
            return i
    raise ValueError(f"metric {name!r} not in spec")


def visibility(
    spec: dict[str, Any],
    *,
    metric: str,
    classifier: str,
) -> list[bool]:
    mi = metric_index(spec, metric)
    ci = classifier_index(spec, classifier)
    vis = [False] * spec["trace_count"]
    base = mi * spec["traces_per_metric"]
    for panel in spec["classifiers"][ci]["panels"]:
        for offset in panel["offsets"]:
            vis[base + offset] = True
    return vis


def layout_patch(
    spec: dict[str, Any],
    *,
    classifier: str,
    annotation_count: int | None = None,
) -> dict[str, Any]:
    """Plotly relayout patch: expand the active classifier, collapse all other axes."""
    ci = classifier_index(spec, classifier)
    active_axes: set[str] = set()
    patch: dict[str, Any] = {}

    dual = spec["panels_per_classifier"] == 2
    for idx, entry in enumerate(spec["classifiers"]):
        show = idx == ci
        for pi, panel in enumerate(entry["panels"]):
            xkey = panel["xkey"]
            ykey = panel["ykey"]
            active_axes.add(xkey)
            active_axes.add(ykey)
            if show:
                if dual:
                    xdom = REAL_DOMAIN if pi == 0 else GEN_DOMAIN
                else:
                    xdom = FULL_X_DOMAIN
                patch[f"{xkey}.domain"] = xdom
                patch[f"{ykey}.domain"] = FULL_Y_DOMAIN
            else:
                patch[f"{xkey}.domain"] = HIDDEN_DOMAIN
                patch[f"{ykey}.domain"] = HIDDEN_DOMAIN

    for key in spec["all_axis_keys"]:
        if key not in active_axes:
            patch[f"{key}.domain"] = HIDDEN_DOMAIN

    n_ann = annotation_count if annotation_count is not None else len(spec["classifiers"]) * 2
    for ann_i in range(n_ann):
        visible = any(
            panel["ann_index"] == ann_i
            for panel in spec["classifiers"][ci]["panels"]
        )
        patch[f"annotations[{ann_i}].visible"] = visible
        if visible:
            patch[f"annotations[{ann_i}].y"] = 1.0

    return patch


def apply_initial_state(
    data: list[dict[str, Any]],
    layout: dict[str, Any],
    spec: dict[str, Any],
) -> None:
    metric = spec["default_metric"]
    classifier = spec["default_classifier"]
    vis = visibility(spec, metric=metric, classifier=classifier)
    for i, trace in enumerate(data):
        trace["visible"] = vis[i]

    relayout = layout_patch(
        spec,
        classifier=classifier,
        annotation_count=len(layout.get("annotations", [])),
    )
    for key, value in relayout.items():
        if key.startswith("annotations["):
            m = re.match(r"annotations\[(\d+)\]\.(\w+)", key)
            if not m:
                continue
            ann_i, field = int(m.group(1)), m.group(2)
            layout["annotations"][ann_i][field] = value
        else:
            axis_key, _, field = key.partition(".")
            layout[axis_key][field] = value

    layout.pop("updatemenus", None)
    layout["height"] = spec["single_height"]
    layout["width"] = spec["width"]
