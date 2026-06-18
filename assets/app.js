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

  function uniqueValues(plots, key) {
    const seen = new Set();
    const out = [];
    for (const p of plots) {
      if (!seen.has(p[key])) {
        seen.add(p[key]);
        out.push(p[key]);
      }
    }
    return out;
  }

  // Plots matching every selected dimension except the ones listed in `ignore`.
  function plotsMatching(ignore) {
    const ignoreSet = new Set(ignore || []);
    return state.manifest.plots.filter((p) =>
      state.dims.every(
        (d) => ignoreSet.has(d.key) || p[d.key] === state.selected[d.key]
      )
    );
  }

  // A value for `key` is available if some plot matches all OTHER selections.
  function availableValues(key) {
    const candidates = plotsMatching([key]);
    return new Set(candidates.map((p) => p[key]));
  }

  function currentPlot() {
    const matches = plotsMatching([]);
    return matches.length === 1 ? matches[0] : null;
  }

  function render() {
    // Repair any selection that became invalid after a change.
    for (const d of state.dims) {
      const avail = availableValues(d.key);
      if (!avail.has(state.selected[d.key])) {
        const firstAvail = [...avail][0];
        if (firstAvail !== undefined) state.selected[d.key] = firstAvail;
      }
    }

    els.selectors.innerHTML = "";
    for (const d of state.dims) {
      const allVals = uniqueValues(state.manifest.plots, d.key);
      const avail = availableValues(d.key);

      const field = document.createElement("div");
      field.className = "field";

      const label = document.createElement("label");
      label.textContent = d.label;
      field.appendChild(label);

      const options = document.createElement("div");
      options.className = "options";

      for (const val of allVals) {
        const btn = document.createElement("button");
        btn.type = "button";
        btn.textContent = val;
        if (val === state.selected[d.key]) btn.classList.add("active");
        if (!avail.has(val)) btn.disabled = true;
        btn.addEventListener("click", () => {
          state.selected[d.key] = val;
          render();
        });
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
      (plot.cohort || plot.model
        ? ` &nbsp;<span style="opacity:.7">(cohort: ${plot.cohort || "-"}, model: ${plot.model || "-"})</span>`
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

    // Default selection from the first plot.
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
