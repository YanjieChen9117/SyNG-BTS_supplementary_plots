# SyNG-BTS Supplementary Plots

An interactive website hosting the ~240 supplementary learning-curve figures that
did not fit in the main [SyNG-BTS](https://github.com/) paper. Visitors pick a
configuration (data type, cancer subtype, normalization, parameters, offline
augmentation) and the matching interactive Plotly figure is displayed.

## How it works

- Each figure is a self-contained Plotly `learning_curve.html`.
- `manifest.json` is a flat index of every figure plus the ordered list of
  values for each selector dimension.
- `index.html` + `assets/` read `manifest.json`, render the selectors, and load
  the chosen figure in an `<iframe>` (so all Plotly hover/zoom interactivity is
  preserved).

## Directory layout

```
.
├── index.html              # the website entry point
├── assets/
│   ├── style.css
│   └── app.js              # builds selectors, loads plots
├── manifest.json           # generated index of all plots (do not edit by hand)
├── generate_manifest.py    # rebuild manifest.json by scanning plots/
├── scripts/
│   └── migrate_batches.py  # one-off importer from the raw batch outputs
└── plots/
    └── <data_type>/<subtype>/<group_label>/<normalization>/<offaug>/<param>/learning_curve.html
```

Each path segment maps directly to a field:

| Segment        | Meaning                          | Selector?            |
|----------------|----------------------------------|----------------------|
| `data_type`    | e.g. `RNA`, `miRNA`              | yes                  |
| `subtype`      | cancer code, e.g. `SKCM`, `KIRP` | yes (Cancer subtype) |
| `group_label`  | classification target            | no (shown as caption)|
| `normalization`| `raw` / `TC` / `DESeq`           | yes                  |
| `offaug`       | offline augmentation: `none`, `AE_head_2`, … | yes (Offline augmentation) |
| `param`        | model config, e.g. `CVAE1-50`    | yes (Parameters)     |

Data types that have no offline augmentation simply use `offaug = none`.

## Adding more plots

1. Place the new `learning_curve.html` files under `plots/` following the layout
   above (only this HTML file is needed; the website ignores any sibling CSV/PNG).
2. Regenerate the index:

   ```bash
   python3 generate_manifest.py
   ```

3. Commit and push. The site updates automatically.

> `scripts/migrate_batches.py` documents how the original `data/learning_curve_output_RNA`
> and `data/learning_curve_output_miRNA` batches were imported. A future batch with a
> different folder convention can be handled by adding a small importer there.

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
