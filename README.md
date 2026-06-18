# SyNG-BTS Supplementary Plots

An interactive website hosting the ~240 supplementary learning-curve figures that
did not fit in the main [SyNG-BTS](https://github.com/) paper. Visitors pick a
configuration (data type, cancer subtype, normalization, parameters, offline
augmentation) and the matching interactive Plotly figure is displayed.

## How it works

- Each figure is a self-contained Plotly `learning_curve.html` living under
  `data/learning_curve_output/cohorts/<group>/<subtype>/<norm>/offaug_<x>/<param>/`.
- `manifest.json` is a flat index of every figure and its dimension values.
- `index.html` + `assets/` read `manifest.json`, render the selectors, and load
  the chosen figure in an `<iframe>` (so all Plotly hover/zoom interactivity is
  preserved).

## Directory layout

```
.
├── index.html              # the website entry point
├── assets/
│   ├── style.css
│   └── app.js              # builds selectors, cross-filters, loads plots
├── manifest.json           # generated index of all plots (do not edit by hand)
├── generate_manifest.py    # rebuild manifest.json by scanning data/
└── data/
    └── learning_curve_output/cohorts/<group>/<subtype>/<norm>/offaug_<x>/<param>/learning_curve.html
```

The `<group>` name is parsed as `<cohort>_<datatype>_<model>`
(e.g. `fivesubtypes_rna_cvae` → cohort=`fivesubtypes`, data type=`RNA`, model=`CVAE`).

## Adding more plots

1. Drop the new `learning_curve.html` files into the directory structure above.
2. Regenerate the index:

   ```bash
   python3 generate_manifest.py
   ```

3. Commit and push. The site updates automatically.

> If a future data source uses a different folder convention, adjust the parsing
> logic in `generate_manifest.py` (the `parse_group` / `parse_offaug` helpers).

## Local preview

GitHub Pages serves static files, and the page fetches `manifest.json`, so you
must preview over HTTP (not `file://`):

```bash
python3 -m http.server 8000
# then open http://localhost:8000
```

## Publishing on GitHub Pages

Repository **Settings → Pages → Build and deployment**:

- **Source:** Deploy from a branch
- **Branch:** `main` (or your default), folder `/ (root)`

The site will be published at `https://<user>.github.io/<repo>/`.
A `.nojekyll` file is included so all folders are served verbatim.
