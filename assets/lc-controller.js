/**
 * Learning-curve figure controller: one classifier + one metric at a time.
 * Expects a JSON spec in #lc-view-spec (injected by add_classifier_menu.py).
 */
(function () {
  "use strict";

  var REAL_DOMAIN = [0, 0.46];
  var GEN_DOMAIN = [0.54, 1];
  var FULL_X = [0, 1];
  var FULL_Y = [0, 1];
  var HIDDEN = [0, 0];

  function classifierIndex(spec, name) {
    for (var i = 0; i < spec.classifiers.length; i++) {
      if (spec.classifiers[i].name === name) return i;
    }
    return 0;
  }

  function metricIndex(spec, name) {
    for (var i = 0; i < spec.metrics.length; i++) {
      if (spec.metrics[i] === name) return i;
    }
    return 0;
  }

  function visibilityArray(spec, metricIdx, classifierIdx) {
    var vis = new Array(spec.trace_count);
    for (var i = 0; i < spec.trace_count; i++) vis[i] = false;
    var base = metricIdx * spec.traces_per_metric;
    var panels = spec.classifiers[classifierIdx].panels;
    for (var p = 0; p < panels.length; p++) {
      var offsets = panels[p].offsets;
      for (var o = 0; o < offsets.length; o++) {
        vis[base + offsets[o]] = true;
      }
    }
    return vis;
  }

  function layoutPatch(spec, classifierIdx) {
    var dual = spec.panels_per_classifier === 2;
    var activeAxes = {};
    var patch = {};
    var ci;
    var idx;
    var entry;
    var pi;
    var panel;
    var xkey;
    var ykey;
    var show;

    for (ci = 0; ci < spec.classifiers.length; ci++) {
      entry = spec.classifiers[ci];
      show = ci === classifierIdx;
      for (pi = 0; pi < entry.panels.length; pi++) {
        panel = entry.panels[pi];
        xkey = panel.xkey;
        ykey = panel.ykey;
        activeAxes[xkey] = true;
        activeAxes[ykey] = true;
        if (show) {
          patch[xkey + ".domain"] = dual ? (pi === 0 ? REAL_DOMAIN : GEN_DOMAIN) : FULL_X;
          patch[ykey + ".domain"] = FULL_Y;
        } else {
          patch[xkey + ".domain"] = HIDDEN;
          patch[ykey + ".domain"] = HIDDEN;
        }
      }
    }

    for (idx = 0; idx < spec.all_axis_keys.length; idx++) {
      xkey = spec.all_axis_keys[idx];
      if (!activeAxes[xkey]) {
        patch[xkey + ".domain"] = HIDDEN;
      }
    }

    var nAnn = spec.annotation_count || spec.classifiers.length * 2;
    var activePanels = spec.classifiers[classifierIdx].panels;
    for (idx = 0; idx < nAnn; idx++) {
      show = false;
      for (pi = 0; pi < activePanels.length; pi++) {
        if (activePanels[pi].ann_index === idx) {
          show = true;
          break;
        }
      }
      patch["annotations[" + idx + "].visible"] = show;
      if (show) patch["annotations[" + idx + "].y"] = 1;
    }

    return patch;
  }

  function init(spec) {
    var plotId = spec.plot_id;
    var clfSelect = document.getElementById("lc-classifier-" + plotId);
    var metricSelect = document.getElementById("lc-metric-" + plotId);
    if (!clfSelect || !metricSelect) return;

    var metricIdx = metricIndex(spec, spec.default_metric);
    var classifierIdx = classifierIndex(spec, spec.default_classifier);

    function applyView() {
      Plotly.restyle(
        plotId,
        { visible: visibilityArray(spec, metricIdx, classifierIdx) }
      );
      Plotly.relayout(plotId, layoutPatch(spec, classifierIdx));
    }

    clfSelect.value = spec.classifiers[classifierIdx].name;
    metricSelect.value = spec.metrics[metricIdx];

    clfSelect.addEventListener("change", function () {
      classifierIdx = classifierIndex(spec, clfSelect.value);
      applyView();
    });
    metricSelect.addEventListener("change", function () {
      metricIdx = metricIndex(spec, metricSelect.value);
      applyView();
    });
  }

  window.LCView = { init: init };
})();
