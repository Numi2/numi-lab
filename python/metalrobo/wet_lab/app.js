"use strict";
const $ = (id) => document.getElementById(id),
  token = window.WET_LAB_TOKEN;
let catalogs = [],
  catalog = null,
  current = null,
  selected = null,
  selectedRegions = new Set(),
  view = "before",
  busy = false,
  painting = false,
  t = 0,
  camera = { scale: 1, x: 0, y: 0 },
  points = [],
  playback = null;
let tissueIndex = null, indexedSpecimen = null, indexedSpatial = null;
let regionalIndex = new Map(), observedIndex = new Map(), geneIndex = new Map();
const molecular = () => catalog?.family === "molecular-perturbation";
const spatial = () => catalog?.family === "spatial-tissue";
async function api(path, body) {
  const r = await fetch("/api/" + path, {
    method: body ? "POST" : "GET",
    headers: { "X-Wet-Lab-Token": token, "Content-Type": "application/json" },
    body: body ? JSON.stringify(body) : undefined,
  });
  const d = await r.json();
  if (!r.ok) throw Error(d.error || "Request failed");
  return d;
}
function status(message, error = false) {
  $("status").textContent = message;
  $("status").classList.toggle("error", error);
}
function text(tag, value, className) {
  const n = document.createElement(tag);
  n.textContent = value;
  if (className) n.className = className;
  return n;
}
function option(id, value, label) {
  const n = text("option", label);
  n.value = value;
  $(id).append(n);
}
function modal(title, node) {
  $("dialog-title").textContent = title;
  $("dialog-content").replaceChildren(
    typeof node === "string" ? text("pre", node) : node,
  );
  $("dialog").showModal();
}
function jsonModal(title, data) {
  modal(title, JSON.stringify(data, null, 2));
}
function controls() {
  document
    .querySelectorAll("button,select,input")
    .forEach((e) => (e.disabled = busy));
  $("reveal").disabled =
    busy ||
    !current ||
    current.revealed ||
    current.observationsAvailable === false;
  for (const id of ["verify", "download", "next"])
    $(id).disabled = busy || !current || (id === "next" && !molecular());
  $("save-template").disabled = busy || !molecular();
  $("compile").disabled = busy || !molecular();
  $("campaign-open").disabled = busy || !molecular();
  $("play").disabled = busy || !spatial() || !current;
  document
    .querySelectorAll("[data-view]")
    .forEach(
      (b) =>
        (b.disabled =
          busy ||
          (!current && b.dataset.view !== "before") ||
          (["observed", "error"].includes(b.dataset.view) &&
            !current?.revealed)),
    );
}
async function action(message, fn) {
  if (busy) return;
  busy = true;
  controls();
  status(message);
  try {
    await fn();
  } catch (e) {
    status(e.message, true);
  } finally {
    busy = false;
    controls();
  }
}
function selection() {
  if (molecular())
    return {
      specimen: $("specimen").value,
      target: $("intervention").value,
      regionIDs: [...selectedRegions].sort(),
      timepoint: $("timepoint").value,
    };
  if (spatial()) return { specimen: $("specimen").value };
  return {
    donor: $("specimen").value,
    hours: Number($("timepoint").value),
    intervention: $("intervention").value,
  };
}
async function setCatalog(c) {
  catalog = c;
  if (c.presentation === "population") { window.learnedLab.unmount(); await window.populationLab.mount(c,catalogs); return; }
  window.populationLab?.unmount();
  if (c.family === "learned-spatial-response") { await window.learnedLab.mount(c,catalogs); return; }
  window.learnedLab.unmount();
  current = null;
  selected = null;
  t = 0;
  view = "before";
  camera = { scale: 1, x: 0, y: 0 };
  selectedRegions = new Set(
    molecular() ? c.specimens[0].regions.map((r) => r.id) : [],
  );
  $("assay").value = c.id;
  $("title").textContent = c.title;
  $("subtitle").textContent = molecular()
    ? "A real spatial specimen. A held-out molecular response."
    : spatial()
      ? "Native mechanics and transport. Numerical evidence remains explicit."
      : "Measured donor RNA. Permanent regression assay.";
  $("specimen").replaceChildren();
  for (const s of c.specimens)
    option("specimen", s.id || s.donor, s.id || s.donor);
  $("intervention").replaceChildren();
  option(
    "intervention",
    molecular() ? c.intervention.target : c.intervention,
    molecular() ? c.intervention.target + " gene knockout" : c.intervention,
  );
  $("timepoint").replaceChildren();
  if (molecular())
    for (const time of c.timepoints)
      option("timepoint", time, "Measured study endpoint");
  else
    option(
      "timepoint",
      spatial() ? "registered" : c.observationHours,
      spatial() ? "Registered sample times" : c.observationHours + " hours",
    );
  $("support").textContent = molecular()
    ? "Knockout assignment comes from measured guide barcodes. Select supported regions; no dose or time interpolation."
    : spatial()
      ? "Registered tracer pulses and native compartment fields."
      : "One measured condition and observation time.";
  $("design-note").textContent = molecular()
    ? "Legacy source-file split. Current reconciliation identifies four animals, including a pooled chip. The original record is preserved; regions and cells are not biological replicates."
    : "Repeated runs are deterministic replays, not biological replicates.";
  $("model-info").textContent = molecular()
    ? "Native context ridge, fixed α=1. All common measured genes enter scoring. Regional aggregate prediction; no cell-specific or mechanistic claim."
    : spatial()
      ? "Matter accepted geometry → native partition transport; one-way coupling."
      : "Native context ridge, no-change, training mean and median.";
  $("limits").textContent = Array.isArray(c.limits)
    ? c.limits.join(" ")
    : c.limits;
  $("source").classList.toggle("hidden", !c.sourceCitation);
  if (c.sourceCitation) {
    const u = new URL(c.sourceCitation);
    $("source").href = u.protocol === "https:" ? u.href : "#";
  }
  $("feature").value = molecular()
    ? "Clu"
    : spatial()
      ? "abstract-tracer"
      : "ISG15";
  $("feature-options").replaceChildren();
  for (const f of c.design?.measurements[0]?.featureIDs || [])
    option("feature-options", f, f);
  renderRegions();
  renderAll();
  controls();
}
function renderRegions() {
  const root = $("regions");
  root.replaceChildren();
  if (!molecular()) {
    root.append(
      text(
        "span",
        spatial() ? "Registered compartment protocol" : "Whole selected donor",
        "small",
      ),
    );
    return;
  }
  for (const r of catalog.specimens[0].regions) {
    const b = text(
      "button",
      r.id.replace("region-", "Region "),
      "chip" + (selectedRegions.has(r.id) ? " on" : ""),
    );
    b.onclick = () => {
      if (selectedRegions.has(r.id)) selectedRegions.delete(r.id);
      else selectedRegions.add(r.id);
      current = null;
      selected = null;
      view = "before";
      renderRegions();
      renderAll();
      controls();
      status(
        "New region selection is a draft. The previous experiment remains in history.",
      );
    };
    root.append(b);
  }
  for (const r of catalog.excludedRegions || []) {
    const b = text(
      "button",
      r.id.replace("region-", "Region ") + " · unavailable",
      "chip unavailable",
    );
    b.onclick = () => jsonModal("Region support", r);
    root.append(b);
  }
}
function specimen() {
  return current?.design || catalog?.design;
}
function featureIndex() { return geneIndex.get($("feature").value) ?? -1; }
function regionResult(id) { return regionalIndex.get(id); }
function regionObservation(id) { return observedIndex.get(id); }
function estimate(r, method = "contextRidge") {
  return r?.estimates.find((e) => e.baseline === method);
}
function measurementValue(entity) {
  const i = tissueIndex?.byID.get(entity.id);
  const v = i === undefined ? NaN : tissueIndex.feature($("feature").value)[i];
  return Number.isFinite(v) ? v : null;
}
function valueAt(p, mode = view) {
  if ($("overlay").value === "uncertainty" && !molecular()) return null;
  if (molecular()) {
    const r = regionResult(p.parentID || p.id),
      j = featureIndex();
    if (!current) return mode === "before" ? measurementValue(p) : null;
    if (!r || j < 0) return null;
    if ($("overlay").value === "uncertainty")
      return Math.abs(
        estimate(r).predictedTreated[j] -
          estimate(r, "meanResponse").predictedTreated[j],
      );
    if (mode === "before") return r.control[j];
    if (mode === "predicted") return estimate(r).predictedTreated[j];
    const o = regionObservation(r.regionID);
    if (!o) return null;
    return mode === "observed" ? o.observed[j] : o.residual[j];
  }
  if (spatial() && current) {
    const i = p.index,
      r = current.spatial,
      control = r.arms[0].samples[t].concentrationsMolPerM3[i],
      pred = r.arms[1].samples[t].concentrationsMolPerM3[i],
      obs =
        current.observationFields?.arms?.[1]?.samples?.[t]
          ?.concentrationsMolPerM3[i];
    if (mode === "before") return control;
    if (mode === "predicted") return pred;
    if (mode === "observed") return obs ?? null;
    return obs == null ? null : pred - obs;
  }
  return null;
}
function evidenceFor(mode = view) {
  if (!current && (!molecular() || mode !== "before")) return "UNAVAILABLE";
  if (["observed", "error"].includes(mode) && !current?.revealed)
    return "UNAVAILABLE";
  if (spatial() && $("overlay").value !== "uncertainty")
    return mode === "error" ? "MODEL INFERENCE" : "HYPOTHESIS";
  if ($("overlay").value === "uncertainty")
    return molecular() && current ? "MODEL INFERENCE" : "UNAVAILABLE";
  if (mode === "observed") return spatial() ? "HYPOTHESIS" : "MEASURED";
  if (mode === "error") return "MODEL INFERENCE";
  if (mode === "predicted")
    return molecular()
      ? "MODEL INFERENCE"
      : spatial()
        ? "HYPOTHESIS"
        : "MODEL INFERENCE";
  return "MEASURED";
}
function buildPoints() {
  if (molecular())
    return (specimen()?.entities || [])
      .filter((e) => e.level === "cell" && e.position)
      .map((e) => ({ ...e, xy: e.position }));
  if (spatial() && current)
    return current.spatial.compartments.map((c, i) => ({
      ...c,
      index: i,
      xy: c.positionMetres,
      level: c.kind,
      parentID: "compartments",
    }));
  return [];
}
function bounds() { return tissueIndex.bounds; }
function refreshIndices() {
  const source = specimen(), fields = current?.spatial;
  if (source !== indexedSpecimen || fields !== indexedSpatial || !tissueIndex) {
    points = buildPoints(); tissueIndex = new TissueIndex(points, source?.measurements[0]);
    indexedSpecimen = source; indexedSpatial = fields;
  }
  regionalIndex = new Map((current?.regionalPredictions || []).map(r => [r.regionID,r]));
  observedIndex = new Map((current?.comparison?.regions || []).map(r => [r.regionID,r]));
  geneIndex = new Map((current?.featureIDs || source?.measurements[0]?.featureIDs || []).map((g,i) => [g,i]));
}
function coordinates(p, w, h, b) {
  const scale = Math.min((w - 70) / b.dx, (h - 75) / b.dy) * camera.scale;
  return [
    (p.xy[0] - b.xmin - b.dx / 2) * scale + w / 2 + camera.x,
    (p.xy[1] - b.ymin - b.dy / 2) * scale + h / 2 + camera.y,
  ];
}
function draw() {
  const canvas = $("tissue"),
    rect = canvas.getBoundingClientRect(),
    w = rect.width,
    h = rect.height,
    dpr = devicePixelRatio || 1;
  if (!w || !h) return;
  canvas.width = w * dpr;
  canvas.height = h * dpr;
  const ctx = canvas.getContext("2d");
  ctx.scale(dpr, dpr);
  ctx.fillStyle = "#102c29";
  ctx.fillRect(0, 0, w, h);
  refreshIndices();
  $("empty-canvas").classList.toggle(
    "hidden",
    points.length > 0 || (!molecular() && !spatial() && current),
  );
  $("empty-canvas").textContent = spatial()
    ? "Seal a registered spatial prediction to inspect its native geometry and fields."
    : "Select a specimen and seal its prediction to inspect molecular readouts.";
  $("layer-badge").textContent = evidenceFor();
  $("scale-label").textContent = molecular()
    ? (current ? "Regional RNA · log1p(CPM)" : "Control cells · raw UMI") +
      " · original chip coordinates"
    : spatial()
      ? "Concentration · mol/m³"
      : "RNA · log1p(CPM)";
  if (!points.length) {
    if (current && !molecular() && !spatial()) drawRNA(ctx, w, h);
    return;
  }
  const b = bounds(),
    values = points.map((p) => valueAt(p)).filter((v) => v != null),
    comparisonScale = TissueIndex.sharedScale(
      ["before", "predicted", "observed"].map(mode => points.map(p => valueAt(p, mode) ?? NaN)),
      [points.map(p => valueAt(p, "error") ?? NaN)]),
    max = view === "error" ? comparisonScale.residual : comparisonScale.max;
  for (const p of points) {
    const [x, y] = coordinates(p, w, h, b);
    p.screen = [x, y];
    const v = valueAt(p),
      background = p.annotations?.role === "geometry-background";
    let color =
      v == null
        ? "#3a5650"
        : view === "error"
          ? v >= 0
            ? `hsl(22 68% ${40 + 35 * (1 - Math.abs(v) / max)}%)`
            : `hsl(189 50% ${40 + 30 * (1 - Math.abs(v) / max)}%)`
          : `hsl(145 ${30 + (25 * v) / max}% ${27 + (52 * v) / max}%)`;
    if ($("overlay").value === "evidence")
      color =
        v == null
          ? "#3a5650"
          : ["predicted", "error"].includes(view)
            ? "#d2b572"
            : "#a6d6b3";
    ctx.beginPath();
    const radius = background
      ? 1.35
      : spatial()
        ? p.kind === "cell"
          ? 4
          : 10
        : 3.8;
    if (p.kind === "extracellular")
      ctx.rect(x - radius, y - radius, radius * 2, radius * 2);
    else ctx.arc(x, y, radius, 0, Math.PI * 2);
    ctx.fillStyle = color;
    ctx.fill();
    if (!background && molecular() && selectedRegions.has(p.parentID)) {
      ctx.strokeStyle = "#d6dec17a";
      ctx.lineWidth = 0.7;
      ctx.stroke();
    }
    if (selected?.id === p.id) {
      ctx.beginPath();
      ctx.arc(x, y, radius + 4, 0, Math.PI * 2);
      ctx.strokeStyle = "#fff3b5";
      ctx.lineWidth = 1.5;
      ctx.stroke();
    }
  }
  ctx.font = "10px -apple-system";
  ctx.fillStyle = "#b6cec0";
  ctx.fillText(
    values.length
      ? view === "error"
        ? `−${max.toPrecision(3)} ← 0 → +${max.toPrecision(3)} · predicted − observed`
        : `${comparisonScale.min.toPrecision(3)} → ${max.toPrecision(3)} · shared comparison scale`
      : "No supported values in this layer",
    16,
    h - 34,
  );
  if (molecular() && current) {
    ctx.fillText(
      "Regional means · not single-cell forecasts",
      16,
      h - 52,
    );
  }
  if ($("overlay").value === "uncertainty" && !molecular()) {
    ctx.fillText("Calibrated uncertainty unavailable", 16, 65);
  }
}
function drawRNA(ctx, w, h) {
  const p = current.prediction,
    e = p.estimates.find((x) => x.baseline === "contextRidge"),
    j = featureIndex();
  let ix =
    j >= 0
      ? [j]
      : current.featureIDs
          .map((_, i) => i)
          .sort(
            (a, b) =>
              Math.abs(e.predictedResponse[b]) -
              Math.abs(e.predictedResponse[a]),
          )
          .slice(0, 12);
  const vals = ix.map((i) =>
    view === "before"
      ? 0
      : view === "predicted"
        ? e.predictedResponse[i]
        : view === "observed"
          ? current.comparison?.observedResponse[i]
          : e.predictedResponse[i] -
            (current.comparison?.observedResponse[i] ?? 0),
  );
  if ($("overlay").value === "uncertainty") {
    ctx.fillStyle = "#b8cfc0";
    ctx.fillText("Calibrated uncertainty unavailable", 20, 50);
    return;
  }
  const max = Math.max(1, ...vals.map(Math.abs));
  ctx.font = "12px -apple-system";
  ix.forEach((i, k) => {
    const x = 55 + (k * (w - 80)) / ix.length,
      y = h / 2 - (vals[k] / max) * h * 0.3;
    ctx.fillStyle = "#9accab";
    ctx.beginPath();
    ctx.arc(x, y, 6, 0, 7);
    ctx.fill();
    ctx.fillStyle = "#b8cfc0";
    ctx.fillText(current.featureIDs[i], x - 20, h - 70);
  });
  ctx.fillStyle = "#b8cfc0";
  ctx.fillText(
    view === "before"
      ? "Control response reference: zero change"
      : view === "error"
        ? "Predicted minus measured RNA response"
        : view === "observed"
          ? "Measured RNA response"
          : "Predicted RNA response",
    20,
    50,
  );
}
function table(headers, rows) {
  const t = document.createElement("table"),
    head = t.createTHead().insertRow();
  for (const h of headers) head.append(text("th", h));
  const b = t.createTBody();
  for (const row of rows) {
    const tr = b.insertRow();
    for (const v of row) tr.insertCell().textContent = v;
  }
  return t;
}
const fmt = (v) => (v == null ? "Unavailable" : Number(v).toPrecision(4));
function inspect(p) {
  selected = p;
  draw();
  $("selection-title").textContent = p.id;
  $("selection-description").textContent = molecular()
    ? `${p.annotations?.condition || "Geometry context"} · ${p.parentID} · ${p.biologicalUnitID}`
    : p.kind + " · native compartment";
  $("selection-evidence").replaceChildren();
  const badge = text(
    "button",
    valueAt(p) == null ? "UNAVAILABLE" : evidenceFor(),
    "evidence-badge inference",
  );
  badge.onclick = () =>
    jsonModal("Evidence at this location", {
      entity: p,
      evidence: p.evidence || current?.spatial?.evidence,
      transition: current?.registration?.plan?.transitions,
      readout:
        "Predictions and comparison are regional expectations; actual single-cell endpoint counts are distinct measurements.",
      modelIdentity: current?.registration?.runtimeIdentity,
      replayArtifact: current?.recordDirectory,
    });
  $("selection-evidence").append(badge);
  const dl = document.createElement("dl");
  for (const [k, v] of [
    ["Selected readout", $("feature").value],
    ["Control reference", fmt(valueAt(p, "before"))],
    ["Predicted", fmt(valueAt(p, "predicted"))],
    ["Observed aggregate", fmt(valueAt(p, "observed"))],
    ["Residual", fmt(valueAt(p, "error"))],
    [
      "Cell endpoint UMI",
      molecular() ? fmt(measurementValue(p)) : "Not an RNA assay",
    ],
  ]) {
    dl.append(text("dt", k), text("dd", v));
  }
  $("local-state").replaceChildren(dl);
  const nearest = points
    .filter((q) => q.id !== p.id)
    .map((q) => ({ q, d: Math.hypot(q.xy[0] - p.xy[0], q.xy[1] - p.xy[1]) }))
    .sort((a, b) => a.d - b.d)
    .slice(0, 5);
  $("neighbors").replaceChildren(
    ...nearest.map((n) => {
      const b = text(
        "button",
        n.q.id + " · distance " + fmt(n.d),
        "quiet small",
      );
      b.onclick = () => inspect(n.q);
      return b;
    }),
  );
  trace(p);
}
function trace(p) {
  const svg = $("trace"),
    ns = "http://www.w3.org/2000/svg";
  svg.replaceChildren();
  let values, labels;
  if (spatial() && current) {
    values = current.spatial.arms[1].samples.map(
      (s) => s.concentrationsMolPerM3[p.index],
    );
    labels = current.spatial.arms[1].samples.map((s) => s.timeSeconds + "s");
    $("trace-note").textContent =
      "Registered intervention field at supported native sample times.";
  } else {
    values = ["before", "predicted", "observed"].map((v) => valueAt(p, v));
    labels = ["Control", "Predicted", "Observed"];
    $("trace-note").textContent =
      "State comparison, not a measured trajectory through time.";
  }
  const max = Math.max(1, ...values.filter((v) => v != null).map(Math.abs));
  values.forEach((v, i) => {
    const x = 40 + (i * 180) / Math.max(1, values.length - 1);
    if (v != null) {
      const n = document.createElementNS(ns, "circle");
      n.setAttribute("cx", x);
      n.setAttribute("cy", 105 - (v / max) * 75);
      n.setAttribute("r", "5");
      n.setAttribute("fill", i === 2 ? "#b67747" : "#3d8b69");
      svg.append(n);
    }
    const label = document.createElementNS(ns, "text");
    label.setAttribute("x", x);
    label.setAttribute("y", "135");
    label.setAttribute("text-anchor", "middle");
    label.setAttribute("font-size", "9");
    label.setAttribute("fill", "#66776f");
    label.textContent = labels[i];
    svg.append(label);
  });
}
function renderEvaluation() {
  $("metrics").replaceChildren();
  $("failures").textContent = "";
  if (!current) {
    $("verdict").textContent = "A prediction comes first.";
    return;
  }
  if (!current.revealed) {
    $("verdict").textContent = "Prediction sealed. Observations hidden.";
    $("evaluation-note").textContent =
      "The model and inputs are fixed. Revealing measurements cannot refit this experiment.";
    return;
  }
  const c = current.comparison;
  if (molecular()) {
    $("verdict").textContent =
      c.verdict === "criterion-contradicted"
        ? "The preregistered criterion was contradicted."
        : "The criterion was met on this specimen.";
    $("evaluation-note").textContent =
      "Held-out mouse, known intervention. All selected regions and common genes enter scoring; this is not zero-shot qualification.";
    $("metrics").append(
      table(
        ["Model", "Response RMSE ↓"],
        c.metrics.map((m) => [m.model, fmt(m.rmse)]),
      ),
    );
    const detail = document.createElement("details");
    detail.append(text("summary", "Gene errors and failure regions"));
    for (const g of c.largestGeneErrors) {
      const b = text(
        "button",
        g.gene + " · RMSE " + fmt(g.rmse),
        "quiet small",
      );
      b.onclick = () => {
        $("feature").value = g.gene;
        view = "error";
        renderAll();
      };
      detail.append(b);
    }
    $("metrics").append(detail);
    $("failures").textContent =
      c.failureClusters.map((f) => f.regionID + ": " + f.reason).join(" · ") +
      " Calibration unavailable: no predictive distribution.";
  } else if (spatial()) {
    $("verdict").textContent = c.interpretation;
    $("evaluation-note").textContent =
      "Numerical-reference comparison does not validate tissue biology.";
    $("metrics").append(
      table(
        ["Arm", "Time (s)", "RMSE"],
        c.metrics.map((m) => [m.arm, m.timeSeconds, fmt(m.rmse)]),
      ),
    );
  } else {
    $("verdict").textContent =
      c.verdict === "beats-both-baselines"
        ? "Lower RNA error than both baselines."
        : "Both baseline comparisons did not improve.";
    $("evaluation-note").textContent =
      "Permanent development-data RNA regression; no general biological prediction claim.";
    $("metrics").append(
      table(
        ["Model", "RMSE", "MAE"],
        c.metrics.map((m) => [m.model, fmt(m.rmse), fmt(m.mae)]),
      ),
    );
  }
}
function renderAll() {
  document
    .querySelectorAll("[data-view]")
    .forEach((b) => b.classList.toggle("selected", b.dataset.view === view));
  const times =
    spatial() && current
      ? current.spatial.arms[1].samples.map((s) => s.timeSeconds)
      : [];
  $("scrubber").max = Math.max(0, times.length - 1);
  $("scrubber").value = t;
  $("time-label").textContent = times.length
    ? times[t] + " seconds"
    : molecular()
      ? "Study endpoint"
      : catalog?.observationHours
        ? catalog.observationHours + " hours"
        : "Registered protocol";
  $("events").replaceChildren();
  let events = molecular()
    ? [
        "Clu gene knockout · barcode assignment",
        "Destructive endpoint · timing not inferred",
      ]
    : spatial()
      ? [
          "Registered extracellular pulse field",
          "Measurements at declared sample times",
        ]
      : ["IFN-beta intervention", "6-hour measured RNA"];
  for (const event of events) $("events").append(text("span", event));
  draw();
  renderEvaluation();
  if (selected) inspect(selected);
  else {
    $("selection-title").textContent = "Explore the specimen";
    $("selection-description").textContent = "Select a cell or region to inspect its molecular state and surroundings.";
    for (const id of ["selection-evidence", "local-state", "neighbors", "trace"]) $(id).replaceChildren();
    $("trace-note").textContent = "";
  }
}
function render(r) {
  if (catalog.id !== r.registration.assay.id)
    setCatalog(catalogs.find((c) => c.id === r.registration.assay.id));
  if (current?.id !== r.id) selected = null;
  current = r;
  $("specimen").value = r.registration.specimen || r.registration.donor;
  if (molecular()) selectedRegions = new Set(r.registration.plan.regionIDs);
  $("feature-options").replaceChildren();
  for (const gene of r.featureIDs || ["abstract-tracer"])
    option("feature-options", gene, gene);
  if (r.featureIDs && !r.featureIDs.includes($("feature").value))
    $("feature").value = r.featureIDs[0];
  view = r.revealed ? "error" : "predicted";
  renderRegions();
  renderAll();
  controls();
}
$("assay").onchange = () => {
  setCatalog(catalogs.find((c) => c.id === $("assay").value));
  status("Assay selected.");
};
$("specimen").onchange = () => {
  current = null;
  selected = null;
  renderAll();
  controls();
};
$("all-regions").onclick = () => {
  selectedRegions = new Set(
    catalog.specimens[0].regions?.map((r) => r.id) || [],
  );
  current = null;
  selected = null;
  view = "before";
  renderRegions();
  renderAll();
  controls();
};
$("paint").onclick = () => {
  painting = !painting;
  $("paint").classList.toggle("paint-active", painting);
  $("canvas-help").textContent = painting
    ? "Click a labelled cell to toggle its supported intervention region"
    : "Scroll to zoom · drag to pan · click to inspect";
};
$("feature").onchange = () => renderAll();
$("overlay").onchange = () => renderAll();
for (const button of document.querySelectorAll("[data-view]"))
  button.onclick = () => {
    view = button.dataset.view;
    renderAll();
  };
$("scrubber").oninput = () => {
  t = Number($("scrubber").value);
  renderAll();
};
$("play").onclick = () => {
  if (playback) {
    clearInterval(playback);
    playback = null;
    return;
  }
  playback = setInterval(() => {
    if (!spatial() || !current) {
      clearInterval(playback);
      playback = null;
      return;
    }
    t = (t + 1) % current.spatial.arms[1].samples.length;
    renderAll();
  }, 650);
};
$("reset-camera").onclick = () => {
  camera = { scale: 1, x: 0, y: 0 };
  draw();
};
let drag = null;
$("tissue").onpointerdown = (e) => {
  drag = { x: e.offsetX, y: e.offsetY, px: camera.x, py: camera.y };
  $("tissue").setPointerCapture(e.pointerId);
};
$("tissue").onpointermove = (e) => {
  if (!drag) return;
  camera.x = drag.px + e.offsetX - drag.x;
  camera.y = drag.py + e.offsetY - drag.y;
  draw();
};
$("tissue").onpointerup = (e) => {
  if (!drag) return;
  const moved = Math.hypot(e.offsetX - drag.x, e.offsetY - drag.y);
  drag = null;
  if (moved > 4) return;
  const rect = $("tissue").getBoundingClientRect(), b = bounds();
  const scale = Math.min((rect.width - 70) / b.dx, (rect.height - 75) / b.dy) * camera.scale;
  const picked = tissueIndex.pick((e.offsetX - rect.width / 2 - camera.x) / scale + b.xmin + b.dx / 2,
    (e.offsetY - rect.height / 2 - camera.y) / scale + b.ymin + b.dy / 2, 24 / scale);
  if (!picked) return;
  const p = {p:picked};
  if (painting && molecular()) {
    const id = p.p.parentID;
    if (!catalog.specimens[0].regions.some((r) => r.id === id)) {
      status(
        "This location lacks an admitted matched control and target group.",
        true,
      );
      return;
    }
    if (selectedRegions.has(id)) selectedRegions.delete(id);
    else selectedRegions.add(id);
    current = null;
    view = "before";
    renderRegions();
    renderAll();
    controls();
  }
  inspect(p.p);
};
$("tissue").onwheel = (e) => {
  e.preventDefault();
  camera.scale = Math.max(
    0.5,
    Math.min(12, camera.scale * Math.exp(-e.deltaY * 0.001)),
  );
  draw();
};
new ResizeObserver(draw).observe($("canvas-wrap"));
$("predict").onclick = () =>
  action("Running native prediction and sealing the experiment…", async () => {
    render(
      await api("predict", { assayID: catalog.id, selection: selection() }),
    );
    status("Prediction sealed. Compare with reality when ready.");
  });
$("reveal").onclick = () =>
  action(
    "Verifying the sealed prediction, then revealing observations…",
    async () => {
      render(await api("reveal", { id: current.id }));
      status(
        "Comparison retained, including failures. Biological qualification has not been conferred.",
      );
    },
  );
$("verify").onclick = () =>
  action("Reconstructing native results and comparison…", async () => {
    await api("verify", { id: current.id });
    status(
      "Exact replay verified against retained inputs and native artifacts.",
    );
  });
$("compile").onclick = () =>
  action("Checking supported experiment boundaries…", async () =>
    jsonModal(
      "Typed experiment plan",
      await api("compile", { assayID: catalog.id, selection: selection() }),
    ),
  );
$("next").onclick = () =>
  action("Ranking unrevealed model disagreement…", async () =>
    jsonModal(
      "Next region · hypothesis",
      await api("propose", { id: current.id }),
    ),
  );
$("download").onclick = () => {
  const u = URL.createObjectURL(
      new Blob([JSON.stringify(current, null, 2)], {
        type: "application/json",
      }),
    ),
    a = document.createElement("a");
  a.href = u;
  a.download = "wet-lab-" + current.id + ".json";
  a.click();
  setTimeout(() => URL.revokeObjectURL(u), 1000);
};
$("evidence-open").onclick = () =>
  jsonModal("Evidence & provenance", {
    states: [
      "MEASURED",
      "SIMULATED — VALIDATED DOMAIN",
      "SIMULATED — OUT OF DISTRIBUTION",
      "MODEL INFERENCE",
      "HYPOTHESIS",
      "UNAVAILABLE",
    ],
    currentLayer: evidenceFor(),
    source: specimen()?.sources || catalog.sourceCitation,
    model:
      current?.registration?.runtimeIdentity || current?.registration?.runtime,
    transitions:
      current?.registration?.plan?.transitions || current?.spatial?.evidence,
    assumptions: catalog.limits,
    heldOutStatus: current?.revealed
      ? "revealed after prediction seal"
      : "not revealed",
    uncertainty: "No calibrated predictive interval for this assay",
    replayArtifact:
      current?.recordDirectory || "Create a sealed experiment first",
  });
$("layer-badge").onclick = () => $("evidence-open").click();
$("history-open").onclick = () =>
  action("Loading retained experiments…", async () => {
    const root = document.createElement("div");
    for (const r of (await api("experiments")).reverse()) {
      const b = text(
        "button",
        (r.donor || r.id.slice(0, 8)) +
          " · " +
          (r.revealed ? "compared" : "sealed") +
          " · " +
          r.id.slice(0, 8),
        "history-row",
      );
      b.disabled = r.status !== "sealed";
      b.onclick = () => {
        $("dialog").close();
        action("Opening retained experiment…", async () => {
          render(await api("experiments/" + r.id));
          status("Opened sealed experiment.");
        });
      };
      root.append(b);
    }
    modal("Experiment history", root);
  });
$("save-template").onclick = () => {
  const box = document.createElement("div"),
    input = document.createElement("input"),
    button = text("button", "Save", "primary");
  input.placeholder = "Template name";
  input.setAttribute("aria-label", "Template name");
  box.append(input, button);
  button.onclick = () => {
    const name = input.value;
    action("Saving typed experiment template…", async () => {
      await api("templates", {
        name,
        assayID: catalog.id,
        selection: selection(),
      });
      $("dialog").close();
      status(
        "Template saved. Predictions and observations are not reused as new evidence.",
      );
    });
  };
  modal("Save experiment template", box);
};
$("templates-open").onclick = () =>
  action("Loading templates…", async () => {
    const box = document.createElement("div");
    for (const r of await api("templates")) {
      const b = text("button", r.name, "history-row");
      b.onclick = () => {
        setCatalog(catalogs.find((c) => c.id === r.assayID));
        selectedRegions = new Set(r.selection.regionIDs);
        $("specimen").value = r.selection.specimen;
        $("intervention").value = r.selection.target;
        $("timepoint").value = r.selection.timepoint;
        renderRegions();
        renderAll();
        $("dialog").close();
        status("Template loaded as a new draft.");
      };
      box.append(b);
    }
    if (!box.children.length) box.append(text("p", "No saved templates yet."));
    modal("Experiment templates", box);
  });
$("arms-open").onclick = () =>
  action("Loading experiment comparisons…", async () => {
    const rows = (await api("experiments")).filter(
        (r) => r.status === "sealed",
      ),
      root = document.createElement("div"),
      select = document.createElement("select");
    select.setAttribute("aria-label", "Comparison experiment");
    for (const r of rows) {
      const o = text(
        "option",
        (r.donor || r.id.slice(0, 8)) + " · " + r.id.slice(0, 8),
      );
      o.value = r.id;
      select.append(o);
    }
    const button = text("button", "Compare selected record", "primary");
    root.append(
      text(
        "p",
        "Compare retained results. Different specimens, modalities or units are never pooled.",
      ),
      select,
      button,
    );
    button.onclick = () =>
      action("Comparing retained arms…", async () => {
        const other = await api("experiments/" + select.value);
        const holder = document.createElement("div");
        if (!current || current.family !== other.family) {
          holder.append(
            text("p", "Select two records from the same assay family."),
          );
        } else {
          holder.append(
            table(
              ["Record", "Specimen", "Revealed", "Verdict"],
              [current, other].map((r) => [
                r.id.slice(0, 8),
                r.registration.specimen || r.registration.donor,
                r.revealed ? "yes" : "no",
                r.comparison?.verdict ||
                  r.comparison?.interpretation ||
                  "not scored",
              ]),
            ),
          );
          holder.append(
            text(
              "p",
              "Each experiment retains its own units, selections, evidence and failures.",
            ),
          );
          if (current.family === "molecular-perturbation") {
            for (const r of [current, other]) {
              const j = r.featureIDs.indexOf($("feature").value);
              holder.append(
                text("h3", r.id.slice(0, 8) + " · " + $("feature").value),
              );
              holder.append(
                table(
                  ["Region", "Control", "Predicted", "Observed", "Residual"],
                  r.regionalPredictions.map((p) => {
                    const o = r.comparison?.regions.find(
                        (x) => x.regionID === p.regionID,
                      ),
                      e = p.estimates.find(
                        (x) => x.baseline === "contextRidge",
                      );
                    return [
                      p.regionID,
                      fmt(p.control[j]),
                      fmt(e.predictedTreated[j]),
                      fmt(o?.observed[j]),
                      fmt(o?.residual[j]),
                    ];
                  }),
                ),
              );
            }
          }
        }
        root.append(holder);
      });
    modal("Comparison workspace", root);
  });
$("benchmark-open").onclick = () =>
  action("Loading hard qualification contract…", async () =>
    jsonModal("Arc 2026 · qualification track", await api("qualification")),
  );
function command() {
  const q = $("command").value.trim();
  if (!q) return;
  const lower = q.toLowerCase();
  if (lower === "predict") {
    $("predict").click();
    return;
  }
  if (lower === "compare") {
    $("reveal").click();
    return;
  }
  if (lower === "replay") {
    $("verify").click();
    return;
  }
  if (lower.startsWith("knockout ")) {
    const target = q.slice(9).trim();
    if (!molecular() || target !== catalog.intervention.target) {
      status(
        "Unsupported target. No binding, pathway or phenotype model will be fabricated.",
        true,
      );
      return;
    }
    $("intervention").value = target;
    status(
      "Supported knockout selected. Region selection defines the typed plan.",
    );
    return;
  }
  if (lower.startsWith("region ")) {
    const id = q.slice(7).trim();
    if (!catalog.specimens[0].regions?.some((r) => r.id === id)) {
      status("Unsupported region.", true);
      return;
    }
    selectedRegions = new Set([id]);
    current = null;
    selected = null;
    view = "before";
    renderRegions();
    renderAll();
    controls();
    return;
  }
  const gene = q.replace(/^gene\s+/i, "");
  if (
    (
      current?.featureIDs ||
      specimen()?.measurements[0]?.featureIDs ||
      []
    ).includes(gene)
  ) {
    $("feature").value = gene;
    renderAll();
    status("Readout selected: " + gene);
    return;
  }
  status(
    "Unsupported command. Use “gene Clu”, “region region-1-0”, “knockout Clu”, predict, compare or replay. Arbitrary dose/time requests require a supported model.",
    true,
  );
}
$("command-run").onclick = command;
$("command").onkeydown = (e) => {
  if (e.key === "Enter") command();
};
document.onkeydown = (e) => {
  if (
    e.key === "/" &&
    !["INPUT", "TEXTAREA"].includes(document.activeElement.tagName)
  ) {
    e.preventDefault();
    $("command").focus();
  }
};
$("dialog-close").onclick = () => $("dialog").close();
(async () => {
  try {
    catalogs = await api("assays");
    catalogs.sort(
      (a, b) =>
        (b.presentation === "population" ? 3 : ({"learned-spatial-response":2,"molecular-perturbation":1}[b.family] || 0)) -
        (a.presentation === "population" ? 3 : ({"learned-spatial-response":2,"molecular-perturbation":1}[a.family] || 0)),
    );
    for (const c of catalogs) option("assay", c.id, c.title);
    const shared = await api("shared");
    await setCatalog(catalogs.find(c=>c.id===shared.selection?.assayID) || catalogs[0]);
    status("Select a supported population, define an objective, then seal a prediction.");
  } catch (e) {
    status(e.message, true);
    $("predict").disabled = true;
  }
})();

$("campaign-open").onclick = () => {
  const root = document.createElement("div"),
    input = document.createElement("textarea"),
    s = selection();
  input.setAttribute("aria-label", "Campaign preregistration JSON");
  input.style.cssText = "width:100%;min-height:280px;font:12px monospace";
  input.value = JSON.stringify(
    {
      hypothesis:
        "The registered predictor improves over both baselines in every selected region.",
      matrix: {
        specimen: [s.specimen],
        target: [s.target],
        regionIDs: [s.regionIDs],
        timepoint: [s.timepoint],
      },
      successCriterion: "adapter-primary-criterion-in-every-arm",
      simulationReplicates: 1,
    },
    null,
    2,
  );
  const run = text("button", "Seal campaign predictions", "primary");
  root.append(
    text(
      "p",
      "Supported selections only. All arm predictions are sealed before any observation reveal. Repeated simulations do not create new biological replicates.",
    ),
    input,
    run,
  );
  run.onclick = () =>
    action("Running preregistered campaign arms…", async () => {
      const r = await api("campaign", {
        assayID: catalog.id,
        request: JSON.parse(input.value),
      });
      root.append(text("pre", JSON.stringify(r, null, 2)));
      const reveal = text("button", "Reveal campaign observations", "primary"),
        verify = text("button", "Replay entire campaign", "secondary");
      reveal.onclick = () =>
        action(
          "Revealing observations after the campaign barrier…",
          async () => {
            const c = await api("campaign/reveal", { id: r.id });
            root.append(text("pre", JSON.stringify(c, null, 2)));
            reveal.disabled = true;
          },
        );
      verify.onclick = () =>
        action("Replaying campaign…", async () =>
          root.append(
            text(
              "pre",
              JSON.stringify(
                await api("campaign/verify", { id: r.id }),
                null,
                2,
              ),
            ),
          ),
        );
      root.append(reveal, verify);
    });
  modal("ExperimentCampaign", root);
};

window.setCatalog = setCatalog;
window.wetLabRenderer = () => catalog?.presentation === "population" ? window.populationLab : window.learnedLab;
window.activateLabModel = async id => { const c=catalogs.find(x=>x.id===id) || await api("catalog?assay="+encodeURIComponent(id)); if ((c.presentation === "population") !== (catalog?.presentation === "population")) await setCatalog(c); return window.wetLabRenderer(); };
