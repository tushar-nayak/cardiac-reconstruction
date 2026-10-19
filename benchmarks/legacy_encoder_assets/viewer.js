async function loadMesh(path) {
  const response = await fetch(path);
  return response.json();
}

function makeTrace(mesh, color) {
  return {
    type: "mesh3d",
    x: mesh.x,
    y: mesh.y,
    z: mesh.z,
    i: mesh.i,
    j: mesh.j,
    k: mesh.k,
    color,
    opacity: 0.72,
    flatshading: false,
    lighting: {
      ambient: 0.46,
      diffuse: 0.78,
      roughness: 0.72,
      specular: 0.18
    },
    lightposition: { x: 120, y: 80, z: 180 },
    hovertemplate: "x=%{x:.1f}<br>y=%{y:.1f}<br>z=%{z:.1f}<extra></extra>"
  };
}

function makeLayout() {
  return {
    margin: { l: 0, r: 0, t: 0, b: 0 },
    paper_bgcolor: "#ffffff",
    plot_bgcolor: "#ffffff",
    scene: {
      aspectmode: "data",
      xaxis: { title: "x", showbackground: false, gridcolor: "#d8dee8" },
      yaxis: { title: "y", showbackground: false, gridcolor: "#d8dee8" },
      zaxis: { title: "z", showbackground: false, gridcolor: "#d8dee8" },
      camera: { eye: { x: 1.45, y: 1.25, z: 0.9 } }
    }
  };
}

async function loadViewer() {
  const [prediction, groundTruth] = await Promise.all([
    loadMesh("assets/reconstruction_mesh.json"),
    loadMesh("assets/ground_truth_mesh.json")
  ]);

  const predTarget = document.getElementById("viewer-pred");
  const gtTarget = document.getElementById("viewer-gt");

  const config = {
    responsive: true,
    displaylogo: false,
    modeBarButtonsToRemove: ["lasso2d", "select2d"]
  };

  await Plotly.newPlot(predTarget, [makeTrace(prediction, "#0057b8")], makeLayout(), config);
  await Plotly.newPlot(gtTarget, [makeTrace(groundTruth, "#111111")], makeLayout(), config);

  let syncingCamera = false;
  const syncCamera = async (target, event) => {
    if (syncingCamera || !event["scene.camera"]) {
      return;
    }
    syncingCamera = true;
    await Plotly.relayout(target, { "scene.camera": event["scene.camera"] });
    syncingCamera = false;
  };

  predTarget.on("plotly_relayout", (event) => {
    syncCamera(gtTarget, event);
  });
  gtTarget.on("plotly_relayout", (event) => {
    syncCamera(predTarget, event);
  });
}

loadViewer().catch((error) => {
  for (const id of ["viewer-pred", "viewer-gt"]) {
    const target = document.getElementById(id);
    target.textContent = `Viewer failed to load: ${error.message}`;
  }
});
