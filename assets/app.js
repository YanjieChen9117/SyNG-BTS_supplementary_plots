(function () {
  "use strict";

  const state = {
    manifest: null,
    dims: [],
    selected: {}, // key -> value
  };

  const els = {
    selectors: document.getElementById("selectors"),
    status: document.getElementById("status"),
    caption: document.getElementById("caption"),
    iframe: document.getElementById("plot"),
    placeholder: document.getElementById("placeholder"),
    footerCount: document.getElementById("footer-count"),
  };

  function dimValues(dim) {
    return dim.values || [];
  }

  // Values reachable for a dimension given the currently selected data type.
  // (Within each data type the configuration grid is complete, so we only need
  // to constrain by data type — this avoids dead-ends when switching types.)
  function availableValues(key) {
    if (key === "data_type") {
      return new Set(state.manifest.plots.map((p) => p.data_type));
    }
    const dt = state.selected.data_type;
    return new Set(
      state.manifest.plots.filter((p) => p.data_type === dt).map((p) => p[key])
    );
  }

  // The single plot matching every current selection (or null).
  function currentPlot() {
    const matches = state.manifest.plots.filter((p) =>
      state.dims.every((d) => p[d.key] === state.selected[d.key])
    );
    return matches.length ? matches[0] : null;
  }

  // Fix dimension `key` to `value`, then snap every other dimension to the
  // nearest existing plot (the one sharing the most current selections).
  function selectValue(key, value) {
    const candidates = state.manifest.plots.filter((p) => p[key] === value);
    if (!candidates.length) return;

    let best = candidates[0];
    let bestScore = -1;
    for (const p of candidates) {
      let score = 0;
      for (const d of state.dims) {
        if (d.key === key) continue;
        if (p[d.key] === state.selected[d.key]) score++;
      }
      if (score > bestScore) {
        bestScore = score;
        best = p;
      }
    }
    for (const d of state.dims) state.selected[d.key] = best[d.key];
    render();
  }

  function render() {
    els.selectors.innerHTML = "";
    for (const d of state.dims) {
      const avail = availableValues(d.key);

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
        if (val === state.selected[d.key]) btn.classList.add("active");
        if (!avail.has(val)) btn.disabled = true;
        btn.addEventListener("click", () => selectValue(d.key, val));
        options.appendChild(btn);
      }

      field.appendChild(options);
      els.selectors.appendChild(field);
    }

    updateViewer();
  }

  function updateViewer() {
    const plot = currentPlot();
    if (!plot) {
      els.status.textContent = "No plot matches this combination.";
      els.iframe.hidden = true;
      els.placeholder.hidden = false;
      els.placeholder.textContent = "No plot available for this combination.";
      els.caption.innerHTML = "";
      return;
    }

    els.status.textContent = "";
    els.caption.innerHTML =
      `<strong>${plot.data_type}</strong> · ${plot.subtype} · ${plot.normalization} · ` +
      `${plot.param} · offaug: ${plot.offaug}` +
      (plot.group_label
        ? ` &nbsp;<span style="opacity:.7">(group label: ${plot.group_label})</span>`
        : "");

    els.placeholder.hidden = false;
    els.placeholder.textContent = "Loading plot…";
    els.iframe.hidden = true;
    els.iframe.onload = () => {
      els.placeholder.hidden = true;
      els.iframe.hidden = false;
    };
    els.iframe.src = plot.path;
  }

  function init(manifest) {
    state.manifest = manifest;
    state.dims = manifest.dimensions;

    if (!manifest.plots.length) {
      els.placeholder.textContent = "manifest.json contains no plots. Run generate_manifest.py.";
      return;
    }

    // Default selection = first plot (a guaranteed-valid combination).
    const first = manifest.plots[0];
    for (const d of state.dims) state.selected[d.key] = first[d.key];

    els.footerCount.textContent = `${manifest.count} plots indexed · plot type: ${manifest.plot_type}`;
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
