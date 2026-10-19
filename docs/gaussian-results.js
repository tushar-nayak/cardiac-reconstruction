"use strict";

const base = "assets/gaussian-reference/";
const subjectSelect = document.getElementById("subject");
const seedSelect = document.getElementById("seed");
const overlay = document.getElementById("overlay");
const axisSelect = document.getElementById("axis");
const sliceSlider = document.getElementById("slice");
const status = document.getElementById("status");
const plotIds = ["mesh-target", "mesh-control", "mesh-candidate"];
const assets = new Map();
let pilot, active, camera = { eye: { x: 1.45, y: 1.25, z: 0.9 } };
let syncing = false;

function setStatus(message, error = false) {
  status.textContent = message;
  status.dataset.error = String(error);
}

async function asset(file) {
  if (!assets.has(file)) {
    assets.set(file, fetch(base + file).then(async (response) => {
      if (!response.ok) throw new Error(`Could not load ${file} (${response.status})`);
      return file.endsWith(".bin") ? new Uint8Array(await response.arrayBuffer()) : response.json();
    }).catch((error) => { assets.delete(file); throw error; }));
  }
  return assets.get(file);
}

function number(value, digits = 4) { return value == null ? "Unavailable" : value.toFixed(digits); }

function tableRow(label, before, after) {
  const row = document.createElement("tr");
  for (const value of [label, before, after]) {
    const cell = document.createElement("td");
    cell.textContent = value;
    row.appendChild(cell);
  }
  return row;
}

function selectedMetrics(pair) {
  const before = pair.control.metrics, after = pair.candidate.metrics;
  const rows = [
    ["Whole-volume Dice", number(before.whole_grid.dice), number(after.whole_grid.dice)],
    ["IoU", number(before.whole_grid.iou), number(after.whole_grid.iou)],
    ["Predicted volume · mL", number(before.predicted_volume_ml, 1), number(after.predicted_volume_ml, 1)],
    ["Target volume · mL", number(before.target_volume_ml, 1), number(after.target_volume_ml, 1)],
    ["Absolute volume error · mL", number(before.absolute_volume_error_ml, 1), number(after.absolute_volume_error_ml, 1)],
    ["Mesh components · prediction / target", `${before.geometry.prediction.connected_components} / ${before.geometry.target.connected_components}`, `${after.geometry.prediction.connected_components} / ${after.geometry.target.connected_components}`],
    ["Surface status", before.geometry.prediction.status.replaceAll("_", " "), after.geometry.prediction.status.replaceAll("_", " ")],
    ["ΔK · predicted − target", number(before.geometry.delta_integrated_gaussian_curvature, 3), number(after.geometry.delta_integrated_gaussian_curvature, 3)],
  ];
  rows.push(["Historical sampled IoU · separate diagnostic", number(pair.control.sampled_iou), number(pair.candidate.sampled_iou)]);
  document.getElementById("selected-metrics").replaceChildren(...rows.map((row) => tableRow(...row)));
}

function meshTrace(mesh, color, opacity = 0.8) {
  return { type: "mesh3d", ...mesh, color, opacity, flatshading: false,
    lighting: { ambient: 0.5, diffuse: 0.8, roughness: 0.75, specular: 0.12 },
    hovertemplate: "x=%{x:.1f} mm<br>y=%{y:.1f} mm<br>z=%{z:.1f} mm<extra></extra>" };
}

async function drawMeshes() {
  if (!active) return;
  if (typeof Plotly === "undefined") throw new Error("The 3D viewer library could not load.");
  const meshes = active.meshes;
  const ranges = ["x", "y", "z"].map((axis) => {
    let lo = Infinity, hi = -Infinity;
    for (const mesh of meshes) for (const value of mesh[axis]) { lo = Math.min(lo, value); hi = Math.max(hi, value); }
    if (!Number.isFinite(lo)) return [-1, 1];
    const pad = Math.max((hi - lo) * 0.08, 1);
    return [lo - pad, hi + pad];
  });
  const layout = {
    margin: { l: 0, r: 0, t: 0, b: 0 }, paper_bgcolor: "#fff", uirevision: "shared-camera",
    scene: { aspectmode: "data", camera,
      ...Object.fromEntries(["x", "y", "z"].map((axis, i) => [axis + "axis",
        { title: axis + " · mm", range: ranges[i], showbackground: false, gridcolor: "#e6ebf1" }])) },
  };
  const colors = ["#26394d", "#888f99", "#0057b8"];
  syncing = true;
  try {
    await Promise.all(plotIds.map((id, i) => {
      const traces = [meshTrace(meshes[i], colors[i])];
      if (i > 0 && overlay.checked) traces.push(meshTrace(meshes[0], "#d95a3a", 0.22));
      return Plotly.react(id, traces, structuredClone(layout), { responsive: true, displaylogo: false });
    }));
  } finally { syncing = false; }
  for (const id of plotIds) {
    const plot = document.getElementById(id);
    if (plot.dataset.cameraBound) continue;
    plot.dataset.cameraBound = "true";
    plot.on("plotly_relayout", async (event) => {
      if (syncing || !event["scene.camera"]) return;
      camera = event["scene.camera"];
      syncing = true;
      try {
        await Promise.all(plotIds.filter((other) => other !== id)
          .map((other) => Plotly.relayout(other, { "scene.camera": camera })));
      } finally { syncing = false; }
    });
  }
}

function pixel(mask, index) { return (mask[index >> 3] >> (index & 7)) & 1; }

function drawSlices() {
  if (!active) return;
  const axis = Number(axisSelect.value), index = Number(sliceSlider.value);
  const [depth, height, width] = active.subject.shape;
  const cols = axis === 2 ? height : width;
  const rows = axis === 0 ? height : depth;
  document.getElementById("slice-label").textContent = `${"zyx"[axis]}=${index} · full-label fit`;
  ["target", "control", "candidate"].forEach((name, variant) => {
    const canvas = document.getElementById("slice-" + name);
    canvas.width = cols;
    canvas.height = rows;
    const context = canvas.getContext("2d"), image = context.createImageData(cols, rows);
    for (let row = 0; row < rows; row++) for (let col = 0; col < cols; col++) {
      const z = axis === 0 ? index : row;
      const y = axis === 0 ? row : axis === 1 ? index : col;
      const x = axis === 2 ? index : col;
      const voxel = (z * height + y) * width + x;
      const target = pixel(active.masks[0], voxel), prediction = pixel(active.masks[variant], voxel);
      const color = target && prediction ? [77, 166, 102] : target ? [230, 76, 51] : prediction ? [51, 102, 230] : [255, 255, 255];
      const offset = (row * cols + col) * 4;
      image.data.set([...color, 255], offset);
    }
    context.putImageData(image, 0, 0);
  });
}

function setSliceAxis() {
  if (!active) return;
  const axis = Number(axisSelect.value);
  sliceSlider.max = active.subject.shape[axis] - 1;
  sliceSlider.value = Math.floor(active.subject.shape[axis] * 0.6);
  drawSlices();
}

async function selectResult() {
  const controls = [subjectSelect, seedSelect, overlay, axisSelect, sliceSlider];
  controls.forEach((element) => { element.disabled = true; });
  setStatus(`Loading ${subjectSelect.value}, seed ${seedSelect.value}…`);
  try {
    const subject = pilot.subjects[subjectSelect.value], pair = subject.seeds[seedSelect.value];
    const variants = [subject.target, pair.control, pair.candidate];
    const [meshes, masks] = await Promise.all([
      Promise.all(variants.map((variant) => asset(variant.mesh))),
      Promise.all(variants.map((variant) => asset(variant.mask))),
    ]);
    active = { subject, meshes, masks };
    selectedMetrics(pair);
    setSliceAxis();
    await drawMeshes();
    setStatus(`${subjectSelect.value} · seed ${seedSelect.value} · all surfaces and slices loaded.`);
  } catch (error) { setStatus(error.message, true); }
  finally { controls.forEach((element) => { element.disabled = false; }); }
}

function populateSeeds() {
  const previous = seedSelect.value;
  seedSelect.replaceChildren(...Object.keys(pilot.subjects[subjectSelect.value].seeds).map((seed) => new Option(seed, seed)));
  if ([...seedSelect.options].some((option) => option.value === previous)) seedSelect.value = previous;
}

async function initialize() {
  pilot = await asset("index.json");
  const before = pilot.control_summary.statistics_over_subject_seed_runs;
  const after = pilot.candidate_summary.statistics_over_subject_seed_runs;
  document.getElementById("mean-dice").textContent = number(after.dice.mean);
  document.getElementById("old-dice").textContent = number(before.dice.mean);
  document.getElementById("mean-iou").textContent = number(after.iou.mean);
  // Volume error is retained per pair rather than in the benchmark summary.
  const pairs = Object.values(pilot.subjects).flatMap((subject) => Object.values(subject.seeds));
  document.getElementById("mean-volume").textContent = number(pairs.reduce((sum, pair) => sum + pair.candidate.metrics.absolute_volume_error_ml, 0) / pairs.length, 1);
  document.getElementById("pair-count").textContent = pairs.length;
  subjectSelect.replaceChildren(...Object.keys(pilot.subjects).map((subject) => new Option(subject.replace("_", " "), subject)));
  populateSeeds();
  subjectSelect.addEventListener("change", () => { populateSeeds(); selectResult(); });
  seedSelect.addEventListener("change", selectResult);
  overlay.addEventListener("change", () => drawMeshes().catch((error) => setStatus(error.message, true)));
  axisSelect.addEventListener("change", setSliceAxis);
  sliceSlider.addEventListener("input", drawSlices);
  await selectResult();
}

initialize().catch((error) => setStatus(error.message, true));
