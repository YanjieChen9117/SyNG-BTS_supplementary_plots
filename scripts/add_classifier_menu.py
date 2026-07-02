#!/usr/bin/env python3
"""Add classifier / metric selectors and show one classifier at a time.

Each figure keeps all original traces. A JSON view spec is inferred from the
existing metric ``updatemenus`` masks. Unused subplots are collapsed via
``domain: [0, 0]``.

Run standalone:

    python3 scripts/add_classifier_menu.py

Validate without writing:

    python3 scripts/add_classifier_menu.py --check
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PLOTS_ROOT = ROOT / "plots"
MARKER = "<!-- lc-view-controller v3 -->"

PLOTLY_CDN = (
    '<script charset="utf-8" src="https://cdn.plot.ly/plotly-3.6.0.min.js" '
    'integrity="sha256-QaOVwtVY0T02VaHrr6pnoHLCwayMJp4O5n4YyaE3rJk=" '
    'crossorigin="anonymous"></script>'
)

TOOLBAR_CSS = """
.lc-root { position: relative; font-family: "Open Sans", verdana, arial, sans-serif; }
.lc-toolbar {
  position: absolute;
  top: 10px;
  z-index: 1001;
  display: flex;
  flex-direction: column;
  gap: 2px;
}
.lc-toolbar-left { left: 10px; }
.lc-toolbar-right { right: 10px; align-items: flex-end; }
.lc-toolbar label {
  font-size: 11px;
  line-height: 1;
  color: #506784;
  font-weight: 600;
}
.lc-toolbar select {
  appearance: none;
  min-width: 108px;
  padding: 5px 28px 5px 10px;
  border: 1px solid #c8d4e3;
  border-radius: 2px;
  background: #fff url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='8' height='4' viewBox='0 0 8 4'%3E%3Cpath fill='%23506784' d='M0 0l4 4 4-4z'/%3E%3C/svg%3E") no-repeat right 10px center;
  color: #2a3f5f;
  font: 12px/1.2 "Open Sans", verdana, arial, sans-serif;
  cursor: pointer;
}
.lc-toolbar select:hover { border-color: #506784; }
.lc-toolbar select:focus { outline: 2px solid #c8d4e3; outline-offset: 1px; }
""".strip()

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lc_view_spec import apply_initial_state, infer_view_spec  # noqa: E402


def parse_plotly_newplot(text: str) -> tuple[str, list, dict, dict]:
    match = re.search(r'Plotly\.newPlot\(\s*"([^"]+)"\s*,', text)
    if not match:
        raise ValueError("Plotly.newPlot(...) not found")

    plot_id = match.group(1)
    pos = match.end()

    def read_json(start: int) -> tuple[object, int]:
        while start < len(text) and text[start] in " \n\t":
            start += 1
        opener = text[start]
        if opener not in "[{":
            raise ValueError(f"expected JSON at {start}, got {opener!r}")
        close = "]" if opener == "[" else "}"
        depth = 0
        for idx in range(start, len(text)):
            ch = text[idx]
            if ch == opener:
                depth += 1
            elif ch == close:
                depth -= 1
                if depth == 0:
                    return json.loads(text[start : idx + 1]), idx + 1
        raise ValueError("unbalanced JSON in Plotly.newPlot")

    data, pos = read_json(pos)
    while pos < len(text) and text[pos] in " ,\n\t":
        pos += 1
    layout, pos = read_json(pos)
    while pos < len(text) and text[pos] in " ,\n\t":
        pos += 1
    config, _ = read_json(pos)
    return plot_id, data, layout, config


def extract_embedded_spec(html: str) -> dict | None:
    match = re.search(
        r'id="lc-view-spec" type="application/json">(.*?)</script>',
        html,
        re.DOTALL,
    )
    if not match:
        return None
    return json.loads(match.group(1))


def controller_src_for(path: Path) -> str:
    depth = len(path.relative_to(ROOT).parent.parts)
    return ("../" * depth) + "assets/lc-controller.js"


def build_toolbar(plot_id: str, spec: dict) -> str:
    clf_options = "".join(
        f'<option value="{c["name"]}">{c["name"]}</option>'
        for c in spec["classifiers"]
    )
    metric_options = "".join(
        f'<option value="{m}">{m}</option>' for m in spec["metrics"]
    )
    return (
        f'<div class="lc-toolbar lc-toolbar-left">'
        f'<label for="lc-classifier-{plot_id}">Classifier</label>'
        f'<select id="lc-classifier-{plot_id}">{clf_options}</select>'
        f"</div>"
        f'<div class="lc-toolbar lc-toolbar-right">'
        f'<label for="lc-metric-{plot_id}">Metric</label>'
        f'<select id="lc-metric-{plot_id}">{metric_options}</select>'
        f"</div>"
    )


def transform_html(html: str, *, path: Path | None = None) -> str:
    if MARKER in html:
        return html

    plot_id, data, layout, config = parse_plotly_newplot(html)
    embedded = extract_embedded_spec(html)
    metrics = embedded.get("metrics") if embedded else None
    spec = infer_view_spec(data, layout, metrics=metrics)
    spec["plot_id"] = plot_id
    apply_initial_state(data, layout, spec)

    if path is None:
        path = PLOTS_ROOT / "learning_curve.html"
    js_src = controller_src_for(path)
    spec_json = json.dumps(spec, separators=(",", ":"), ensure_ascii=False)
    data_json = json.dumps(data, separators=(",", ":"), ensure_ascii=False)
    layout_json = json.dumps(layout, separators=(",", ":"), ensure_ascii=False)
    config_json = json.dumps(config, separators=(",", ":"), ensure_ascii=False)
    height = spec["single_height"]
    width = spec["width"]

    return (
        "<html>\n"
        "<head><meta charset=\"utf-8\" />"
        f"<style>{TOOLBAR_CSS}</style></head>\n"
        "<body>\n"
        f"{MARKER}\n"
        f'<div class="lc-root" style="height:{height}px; width:{width}px;">\n'
        "        <script>window.PlotlyConfig = {MathJaxConfig: 'local'};</script>\n"
        f"        {PLOTLY_CDN}\n"
        f'        <script id="lc-view-spec" type="application/json">{spec_json}</script>\n'
        f'        <script src="{js_src}"></script>\n'
        f"        {build_toolbar(plot_id, spec)}"
        f'        <div id="{plot_id}" class="plotly-graph-div" '
        f'style="height:100%; width:100%;"></div>\n'
        "            <script>"
        "                window.PLOTLYENV=window.PLOTLYENV || {};"
        f'                if (document.getElementById("{plot_id}")) {{'
        "                    Plotly.newPlot("
        f'                        "{plot_id}",'
        f"                        {data_json},"
        f"                        {layout_json},"
        f"                        {config_json}"
        "                    ).then(function () {"
        "                        var spec = JSON.parse("
        '                            document.getElementById("lc-view-spec").textContent'
        "                        );"
        "                        window.LCView.init(spec);"
        "                    });"
        "                };"
        "            </script>        </div>\n"
        "</body>\n"
        "</html>\n"
    )


def check_all() -> int:
    errors: list[str] = []
    paths = sorted(PLOTS_ROOT.rglob("learning_curve.html"))
    for path in paths:
        try:
            _, data, layout, _ = parse_plotly_newplot(path.read_text(encoding="utf-8"))
            spec = infer_view_spec(data, layout)
            if spec["panels_per_classifier"] != 2:
                errors.append(
                    f"{path.relative_to(ROOT)}: expected Real+Generated panels, "
                    f"got {spec['panels_per_classifier']}"
                )
        except ValueError as exc:
            errors.append(f"{path.relative_to(ROOT)}: {exc}")

    print(f"Checked {len(paths)} figures.")
    if errors:
        print(f"\nErrors ({len(errors)}):")
        for msg in errors:
            print(f"  {msg}")
        return 1
    print("All figures have dual Real+Generated panels.")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--check",
        action="store_true",
        help="validate inference on all figures without writing",
    )
    args = parser.parse_args()
    if args.check:
        raise SystemExit(check_all())

    changed = 0
    skipped = 0
    errors: list[str] = []

    for path in sorted(PLOTS_ROOT.rglob("learning_curve.html")):
        text = path.read_text(encoding="utf-8")
        if MARKER in text:
            skipped += 1
            continue
        try:
            new = transform_html(text, path=path)
        except ValueError as exc:
            errors.append(f"{path.relative_to(ROOT)}: {exc}")
            continue
        path.write_text(new, encoding="utf-8")
        changed += 1

    print(f"Updated {changed} figure(s), skipped {skipped} already transformed.")
    if errors:
        print(f"\nErrors ({len(errors)}) — left unchanged:")
        for msg in errors:
            print(f"  {msg}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
