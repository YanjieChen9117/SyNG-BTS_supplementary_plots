(function () {
  "use strict";

  const state = {
    manifest: null,
    dims: [],
    selected: {}, // key -> value (or undefined when not selected)
    currentSrc: null,
  };

  // Configuration shown when the page first loads.
  const DEFAULT_SELECTION = {
    data_type: "miRNA",
    subtype: "BLCA",
    normalization: "none",
    param: "CVAE1-5",
    offaug: "none",
  };

  const els = {
    selectors: document.getElementById("selectors"),
    status: document.getElementById("status"),
    caption: document.getElementById("caption"),
    iframe: document.getElementById("plot"),
    placeholder: document.getElementById("placeholder"),
    footerCount: document.getElementById("footer-count"),
    info: document.getElementById("dataset-info"),
    infoSampleSize: document.getElementById("info-sample-size"),
    infoMarkerDim: document.getElementById("info-marker-dim"),
    resetBtn: document.getElementById("reset-btn"),
  };

  function resetSelection() {
    for (const d of state.dims) state.selected[d.key] = undefined;
    render();
  }

  function dimValues(dim) {
    return dim.values || [];
  }

  function selectionCount() {
    return state.dims.filter((d) => state.selected[d.key] != null).length;
  }

  // Values reachable for `key` given every OTHER currently-selected dimension.
  // Unselected dimensions act as wildcards, so the user can never get stuck:
  // any conflicting choice can be cleared to free up other options.
  function availableValues(key) {
    const result = new Set();
    for (const p of state.manifest.plots) {
      let ok = true;
      for (const d of state.dims) {
        if (d.key === key) continue;
        const sel = state.selected[d.key];
        if (sel != null && p[d.key] !== sel) {
          ok = false;
          break;
        }
      }
      if (ok) result.add(p[key]);
    }
    return result;
  }

  // The plot matching the full selection, only when all dimensions are chosen.
  function currentPlot() {
    if (selectionCount() !== state.dims.length) return null;
    return (
      state.manifest.plots.find((p) =>
        state.dims.every((d) => p[d.key] === state.selected[d.key])
      ) || null
    );
  }

  // Match a plot against DEFAULT_SELECTION; normalization accepts raw/none alias.
  function plotMatchesDefaults(plot) {
    return state.dims.every((d) => {
      const want = DEFAULT_SELECTION[d.key];
      if (want == null) return true;
      const got = plot[d.key];
      if (d.key === "normalization") {
        return got === want || (want === "none" && got === "raw") || (want === "raw" && got === "none");
      }
      return got === want;
    });
  }

  function applyDefaultSelection() {
    const plot = state.manifest.plots.find(plotMatchesDefaults);
    if (!plot) return false;
    for (const d of state.dims) state.selected[d.key] = plot[d.key];
    return true;
  }

  function toggleValue(key, value) {
    state.selected[key] = state.selected[key] === value ? undefined : value;
    render();
  }

  function render() {
    els.selectors.innerHTML = "";
    for (const d of state.dims) {
      const avail = availableValues(d.key);
      const selectedVal = state.selected[d.key];

      const field = document.createElement("div");
      field.className = "field";

      const label = document.createElement("label");
      label.textContent = d.label;
      field.appendChild(label);

      const options = document.createElement("div");
      options.className = "options";

      for (const val of dimValues(d)) {
        const btn = document.createElement("button");
        btn.type = "button";
        btn.textContent = val;
        const isActive = val === selectedVal;
        if (isActive) btn.classList.add("active");
        // Never disable the active value (so it can always be cleared).
        if (!isActive && !avail.has(val)) btn.disabled = true;
        btn.addEventListener("click", () => toggleValue(d.key, val));
        options.appendChild(btn);
      }

      field.appendChild(options);
      els.selectors.appendChild(field);
    }

    updateViewer();
  }

  function updateViewer() {
    const chosen = selectionCount();
    const total = state.dims.length;
    const plot = currentPlot();

    if (!plot) {
      state.currentSrc = null;
      els.iframe.hidden = true;
      els.iframe.removeAttribute("src");
      els.placeholder.hidden = false;
      els.caption.innerHTML = "";
      els.placeholder.textContent =
        chosen < total
          ? `Select all ${total} options to display a plot (${chosen}/${total} chosen).`
          : "No plot available for this combination.";
      els.status.textContent = "";
      els.info.hidden = true;
      return;
    }

    els.status.textContent = "";
    els.caption.innerHTML =
      `<strong>${plot.data_type}</strong> · ${plot.subtype} · ${plot.normalization} · ` +
      `${plot.param} · offaug: ${plot.offaug}` +
      (plot.group_label
        ? ` &nbsp;<span style="opacity:.7">(group label: ${plot.group_label})</span>`
        : "");

    // Only (re)load the iframe when the target actually changes.
    if (state.currentSrc !== plot.path) {
      state.currentSrc = plot.path;
      els.iframe.src = plot.path;
    }
    els.placeholder.hidden = true;
    els.iframe.hidden = false;

    // Dataset difficulty panel (only meaningful once the full set is chosen).
    els.infoSampleSize.textContent = plot.sample_size != null ? plot.sample_size : "—";
    els.infoMarkerDim.textContent = plot.marker_dim != null ? plot.marker_dim : "—";
    els.info.hidden = false;
  }

  function init(manifest) {
    state.manifest = manifest;
    state.dims = manifest.dimensions;

    if (!manifest.plots.length) {
      els.placeholder.textContent = "manifest.json contains no plots. Run generate_manifest.py.";
      return;
    }

    // Apply the default configuration by finding a matching plot so every
    // dimension (including normalization) is set to a valid combination.
    if (!applyDefaultSelection()) {
      for (const d of state.dims) {
        const def = DEFAULT_SELECTION[d.key];
        state.selected[d.key] =
          def != null && dimValues(d).includes(def) ? def : undefined;
      }
    }

    els.footerCount.textContent = `${manifest.count} plots indexed`;
    els.resetBtn.addEventListener("click", resetSelection);
    render();
  }

  fetch("manifest.json", { cache: "no-cache" })
    .then((r) => {
      if (!r.ok) throw new Error("HTTP " + r.status);
      return r.json();
    })
    .then(init)
    .catch((err) => {
      els.placeholder.textContent =
        "Failed to load manifest.json (" + err.message + "). " +
        "If viewing locally, serve over http (e.g. `python3 -m http.server`).";
    });
})();
