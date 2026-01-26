import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { CSS2DObject, CSS2DRenderer } from "three/addons/renderers/CSS2DRenderer.js";

const elements = {
  scenePath: document.getElementById("scenePath"),
  load: document.getElementById("load"),
  meta: document.getElementById("meta"),
  smplMeshPath: document.getElementById("smplMeshPath"),
  showMesh: document.getElementById("showMesh"),
  showSkeleton: document.getElementById("showSkeleton"),
  showBodies: document.getElementById("showBodies"),
  showMarkers: document.getElementById("showMarkers"),
  showLabels: document.getElementById("showLabels"),
  showSmpl: document.getElementById("showSmpl"),
  showGrid: document.getElementById("showGrid"),
  showAxes: document.getElementById("showAxes"),
  markerSize: document.getElementById("markerSize"),
  markerFilter: document.getElementById("markerFilter"),
  markerList: document.getElementById("markerList"),
  showAllMarkers: document.getElementById("showAllMarkers"),
  hideAllMarkers: document.getElementById("hideAllMarkers"),
  selectedMarker: document.getElementById("selectedMarker"),
  viewer: document.getElementById("viewer"),
  startPick: document.getElementById("startPick"),
  savePick: document.getElementById("savePick"),
  cancelPick: document.getElementById("cancelPick"),
  pickStatus: document.getElementById("pickStatus"),
  exportPicks: document.getElementById("exportPicks"),
  pickNote: document.getElementById("pickNote"),
  previewSize: document.getElementById("previewSize"),
  savedSize: document.getElementById("savedSize"),
};

const geometryCache = new Map();

function getUrlParams() {
  const params = new URLSearchParams(window.location.search);
  const scene = params.get("scene");
  if (scene) elements.scenePath.value = scene;
  const smpl = params.get("smplMeshPath");
  if (smpl && elements.smplMeshPath) elements.smplMeshPath.value = smpl;
}

function setMeta(text) {
  elements.meta.textContent = text;
}

function makeRenderer(container) {
  const renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setPixelRatio(window.devicePixelRatio || 1);
  renderer.setSize(container.clientWidth, container.clientHeight);
  container.replaceChildren(renderer.domElement);
  return renderer;
}

function makeLabelRenderer(container) {
  const labelRenderer = new CSS2DRenderer();
  labelRenderer.setSize(container.clientWidth, container.clientHeight);
  labelRenderer.domElement.style.position = "absolute";
  labelRenderer.domElement.style.top = "0";
  labelRenderer.domElement.style.left = "0";
  labelRenderer.domElement.style.pointerEvents = "none";
  container.appendChild(labelRenderer.domElement);
  return labelRenderer;
}

function resize(renderer, labelRenderer, camera, container) {
  const w = container.clientWidth;
  const h = container.clientHeight;
  if (!w || !h) return;
  renderer.setSize(w, h);
  labelRenderer.setSize(w, h);
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
}

async function loadSceneJson(url) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`Failed to fetch scene JSON: ${res.status} ${res.statusText} (${url})`);
  return res.json();
}

function parseNumbers(text, asInt = false) {
  const parts = text.trim().split(/\s+/);
  if (!parts.length) return asInt ? new Int32Array(0) : new Float32Array(0);
  if (asInt) {
    const out = new Int32Array(parts.length);
    for (let i = 0; i < parts.length; i++) out[i] = Number(parts[i]);
    return out;
  }
  const out = new Float32Array(parts.length);
  for (let i = 0; i < parts.length; i++) out[i] = Number(parts[i]);
  return out;
}

function parseVtpAscii(vtpText) {
  const xml = new DOMParser().parseFromString(vtpText, "text/xml");
  const pointsNode = xml.querySelector("Points DataArray");
  const connNode = xml.querySelector("Polys DataArray[Name='connectivity']");
  const normalsNode = xml.querySelector("PointData DataArray[Name='Normals']");

  if (!pointsNode || !connNode) {
    throw new Error("Failed to parse VTP: missing Points or connectivity");
  }

  const positions = parseNumbers(pointsNode.textContent, false);
  const connectivity = parseNumbers(connNode.textContent, true);

  let normals = null;
  if (normalsNode) {
    const n = parseNumbers(normalsNode.textContent, false);
    if (n.length === positions.length) normals = n;
  }

  let maxIndex = 0;
  for (let i = 0; i < connectivity.length; i++) maxIndex = Math.max(maxIndex, connectivity[i]);
  const index = maxIndex > 65535 ? new Uint32Array(connectivity) : new Uint16Array(connectivity);

  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));
  if (normals) geometry.setAttribute("normal", new THREE.BufferAttribute(normals, 3));
  geometry.setIndex(new THREE.BufferAttribute(index, 1));
  if (!normals) geometry.computeVertexNormals();
  geometry.computeBoundingSphere();
  return geometry;
}

async function loadVtpGeometry(filename) {
  const base = String(filename || "").split(/[\\/]/).pop();
  if (!base) throw new Error(`Invalid VTP filename: ${filename}`);

  if (geometryCache.has(base)) return geometryCache.get(base);

  const promise = (async () => {
    const url = `./Geometry/${encodeURIComponent(base)}`;
    const res = await fetch(url);
    if (!res.ok) {
      throw new Error(`Failed to fetch VTP: ${res.status} ${res.statusText} (${url})`);
    }
    const text = await res.text();
    return parseVtpAscii(text);
  })();

  geometryCache.set(base, promise);
  return promise;
}

function computeBounds(points) {
  const box = new THREE.Box3();
  for (const p of points) box.expandByPoint(new THREE.Vector3(p[0], p[1], p[2]));
  return box;
}

function fitCameraToBox(camera, controls, box) {
  const center = box.getCenter(new THREE.Vector3());
  const size = box.getSize(new THREE.Vector3());
  const radius = Math.max(size.x, size.y, size.z) * 0.6 + 0.2;

  controls.target.copy(center);
  controls.update();

  const dir = new THREE.Vector3(1, 0.6, 1).normalize();
  camera.position.copy(center.clone().addScaledVector(dir, radius * 3.0));
  camera.near = Math.max(0.01, radius / 200);
  camera.far = Math.max(1000, radius * 50);
  camera.updateProjectionMatrix();
}

function buildScene(container) {
  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0x0b0f14);

  const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 1000);
  camera.position.set(2.5, 1.6, 2.5);

  const renderer = makeRenderer(container);
  const labelRenderer = makeLabelRenderer(container);

  const controls = new OrbitControls(camera, renderer.domElement);
  controls.target.set(0.0, 1.0, 0.0);
  controls.update();

  const light1 = new THREE.DirectionalLight(0xffffff, 1.05);
  light1.position.set(3, 6, 2);
  scene.add(light1);
  scene.add(new THREE.AmbientLight(0xffffff, 0.35));

  const grid = new THREE.GridHelper(10, 20, 0x2b3340, 0x1b2230);
  grid.name = "grid";
  scene.add(grid);

  const axes = new THREE.AxesHelper(0.3);
  axes.name = "axes";
  scene.add(axes);

  const root = new THREE.Group();
  root.name = "root";
  scene.add(root);

  const modelGroup = new THREE.Group();
  modelGroup.name = "model";
  root.add(modelGroup);

  const overlayGroup = new THREE.Group();
  overlayGroup.name = "overlay";
  root.add(overlayGroup);
  const manualGroup = new THREE.Group();
  manualGroup.name = "manual";
  overlayGroup.add(manualGroup);
  const smplGroup = new THREE.Group();
  smplGroup.name = "smpl";
  overlayGroup.add(smplGroup);

  window.addEventListener("resize", () => resize(renderer, labelRenderer, camera, container));
  resize(renderer, labelRenderer, camera, container);

  return { scene, camera, renderer, labelRenderer, controls, grid, axes, root, modelGroup, overlayGroup, manualGroup, smplGroup };
}

function buildSkeleton(root, bodies, edges) {
  const group = new THREE.Group();
  group.name = "skeleton";

  const validEdges = edges.filter(([a, b]) => bodies.has(a) && bodies.has(b));

  const positions = new Float32Array(validEdges.length * 2 * 3);
  for (let i = 0; i < validEdges.length; i++) {
    const [a, b] = validEdges[i];
    const pa = bodies.get(a);
    const pb = bodies.get(b);
    const base = i * 2 * 3;
    positions[base + 0] = pa[0];
    positions[base + 1] = pa[1];
    positions[base + 2] = pa[2];
    positions[base + 3] = pb[0];
    positions[base + 4] = pb[1];
    positions[base + 5] = pb[2];
  }

  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));
  const material = new THREE.LineBasicMaterial({ color: 0x66c2ff });
  const lines = new THREE.LineSegments(geometry, material);
  group.add(lines);

  root.add(group);
  return { group };
}

function buildBodyPoints(root, bodies) {
  const group = new THREE.Group();
  group.name = "bodies";

  const names = [...bodies.keys()].sort();
  const positions = new Float32Array(names.length * 3);
  for (let i = 0; i < names.length; i++) {
    const p = bodies.get(names[i]);
    positions[i * 3 + 0] = p[0];
    positions[i * 3 + 1] = p[1];
    positions[i * 3 + 2] = p[2];
  }
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));
  const material = new THREE.PointsMaterial({ color: 0xffcc66, size: 0.02 });
  const points = new THREE.Points(geometry, material);
  group.add(points);

  root.add(group);
  return { group, pointsMaterial: material };
}

async function loadSmplMesh(path, baseMeta = "") {
  if (!path) return;
  const url = `${path}${path.includes("?") ? "&" : "?"}t=${Date.now()}`; // bust cache
  const res = await fetch(url, { cache: "no-store" });
  if (!res.ok) throw new Error(`Failed to fetch SMPL mesh: ${res.status} ${res.statusText} (${path})`);
  const data = await res.json();
  const verts = new Float32Array(data.vertices || []);
  const faces = new Uint32Array(data.faces || []);
  if (!verts.length || !faces.length) throw new Error("SMPL mesh JSON missing vertices/faces");
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute("position", new THREE.BufferAttribute(verts, 3));
  geometry.setIndex(new THREE.BufferAttribute(faces, 1));
  geometry.computeVertexNormals();
  geometry.computeBoundingSphere();
  geometry.computeBoundingBox();
  const material = new THREE.MeshStandardMaterial({
    color: 0x915eff, // transparent purple fill
    wireframe: false,
    opacity: 0.25,
    transparent: true,
    depthWrite: false,
    roughness: 0.7,
    metalness: 0.05,
  });
  const mesh = new THREE.Mesh(geometry, material);
  mesh.name = "smpl_mesh";
  mesh.frustumCulled = false;
  mesh.userData.meshType = "smpl";

  // Point cloud overlay to help visualize vertices
  const ptsMat = new THREE.PointsMaterial({
    color: 0x00ff99,
    size: 0.006,
    transparent: true,
    opacity: 0.9,
    depthWrite: false,
    sizeAttenuation: true,
  });
  const points = new THREE.Points(geometry, ptsMat);
  points.name = "smpl_points";
  points.userData.meshType = "smpl";

  const group = new THREE.Group();
  group.name = "smpl_container";
  const wire = mesh.clone();
  wire.material = new THREE.MeshBasicMaterial({ color: 0x55ffcc, wireframe: true, transparent: true, opacity: 0.5, depthWrite: false });
  group.add(mesh);
  group.add(wire);
  group.add(points);
  return { group, box: geometry.boundingBox?.clone() };
}

async function ensureSmplMesh(viz, current, baseMeta = "") {
  const smplPath = elements.smplMeshPath?.value.trim();
  if (!elements.showSmpl?.checked || !smplPath) return;
  setMeta(`${baseMeta}\nSMPL: Loading...`);
  try {
    const result = await loadSmplMesh(smplPath, baseMeta);
    const smplMesh = result?.group;
    const smplBox = result?.box;
    viz.smplGroup.clear();
    if (smplMesh) viz.smplGroup.add(smplMesh);
    viz.smplGroup.visible = true;
    current.smpl = { mesh: smplMesh, path: smplPath, box: smplBox };
    if (smplBox) {
      const box = smplBox.clone();
      fitCameraToBox(viz.camera, viz.controls, box);
    }
    setMeta(`${baseMeta}\nSMPL: Loaded`);
  } catch (e) {
    setMeta(`${baseMeta}\nSMPL: Failed ${e}`);
  }
}

function markerColor(name) {
  if (name.toLowerCase().endsWith("_smpl")) return 0xff3333;
  if (name.toLowerCase().endsWith("_opencap")) return 0x3399ff;
  // Simple heuristic coloring by region name to help scanning.
  const n = name.toLowerCase();
  if (n.includes("asis") || n.includes("psis") || n.includes("hjc") || n.includes("c7")) return 0x8bd3ff;
  if (n.includes("thigh") || n.includes("knee") || n.includes("ankle") || n.includes("toe") || n.includes("calc") || n.includes("meta"))
    return 0x7cffa7;
  if (n.includes("shoulder") || n.includes("elbow") || n.includes("wrist") || n.includes("sh")) return 0xffc27a;
  return 0xd6d6d6;
}

function buildMarkers(root, markers, markerRadius, showLabels) {
  const group = new THREE.Group();
  group.name = "markers";

  const geometry = new THREE.SphereGeometry(1, 12, 10);
  const markerMap = new Map();

  for (const { name, position, color } of markers) {
    let matColor;
    if (Array.isArray(color) && color.length >= 3) {
      matColor = new THREE.Color(color[0], color[1], color[2]);
    } else {
      matColor = new THREE.Color(markerColor(name));
    }
    const material = new THREE.MeshStandardMaterial({
      color: matColor,
      roughness: 0.65,
      metalness: 0.05,
      emissive: 0x000000,
    });
    const mesh = new THREE.Mesh(geometry, material);
    mesh.position.set(position[0], position[1], position[2]);
    mesh.scale.setScalar(markerRadius);
    mesh.name = `marker:${name}`;
    mesh.userData.markerName = name;

    const labelEl = document.createElement("div");
    labelEl.className = "label";
    labelEl.textContent = name;
    const label = new CSS2DObject(labelEl);
    label.position.set(0, markerRadius * 1.2, 0);
    label.visible = Boolean(showLabels);
    mesh.add(label);

    group.add(mesh);
    markerMap.set(name, { mesh, label, material });
  }

  root.add(group);
  return { group, markerMap };
}

function renderMarkerList(markerMap) {
  elements.markerList.replaceChildren();

  const names = [...markerMap.keys()].sort();
  for (const name of names) {
    const item = document.createElement("label");
    item.className = "marker-item";
    item.dataset.name = name;

    const checkbox = document.createElement("input");
    checkbox.type = "checkbox";
    checkbox.checked = true;
    checkbox.addEventListener("change", () => {
      const entry = markerMap.get(name);
      if (!entry) return;
      entry.mesh.visible = checkbox.checked;
    });

    const code = document.createElement("code");
    code.textContent = name;

    item.appendChild(checkbox);
    item.appendChild(code);
    elements.markerList.appendChild(item);
  }
}

function applyMarkerFilter() {
  const q = elements.markerFilter.value.trim().toLowerCase();
  const items = elements.markerList.querySelectorAll(".marker-item");
  for (const item of items) {
    const name = (item.dataset.name || "").toLowerCase();
    item.style.display = !q || name.includes(q) ? "" : "none";
  }
}

async function start() {
  getUrlParams();

  const viz = buildScene(elements.viewer);
  const raycaster = new THREE.Raycaster();
  const pointer = new THREE.Vector2();
  const clickState = { down: null };
  let previewMesh = null;

  let current = {
    data: null,
    skeleton: null,
    bodies: null,
    markers: null,
    markerMap: null,
    selectedName: null,
    manualSelecting: false,
    pendingPick: null,
    picks: [],
    pickId: 1,
    smpl: null,
  };
  updatePickUi();

  function resetManualState() {
    current.manualSelecting = false;
    current.pendingPick = null;
    current.picks = [];
    current.pickId = 1;
    if (viz.manualGroup) viz.manualGroup.clear();
    updatePickUi();
  }

  function updatePickUi() {
    const selText = current.pendingPick
      ? `Pending save: ${current.pendingPick.type === "marker" ? current.pendingPick.name : `${current.pendingPick.geom}#${current.pendingPick.vertex}`}\nCoords: ${current.pendingPick.position
          .map((v) => v.toFixed(4))
          .join(", ")}\nNotes: ${(elements.pickNote?.value || "").trim() || "(none)"}`
      : current.manualSelecting
        ? "Pick mode on: click markers or mesh vertices"
        : "Not selected";
    if (elements.pickStatus) elements.pickStatus.textContent = selText;
    if (elements.savePick) elements.savePick.disabled = !current.pendingPick;
    if (elements.cancelPick) elements.cancelPick.disabled = !current.pendingPick && !current.manualSelecting;
    if (elements.exportPicks) elements.exportPicks.disabled = !current.picks.length;
  }

function showPreview(position) {
  if (!viz.manualGroup) return;
  if (!previewMesh) {
    const g = new THREE.SphereGeometry(1, 10, 8);
    const mat = new THREE.MeshStandardMaterial({ color: 0xffff66, emissive: 0x555500, roughness: 0.45, metalness: 0.1, transparent: true, opacity: 0.9 });
    previewMesh = new THREE.Mesh(g, mat);
    const r = Number(elements.previewSize?.value || "0.015");
    previewMesh.scale.setScalar(r);
    viz.manualGroup.add(previewMesh);
  }
  const r = Number(elements.previewSize?.value || "0.015");
  previewMesh.scale.setScalar(r);
  previewMesh.position.set(position[0], position[1], position[2]);
  previewMesh.visible = true;
}

  function updateSelectedUi() {
    if (!elements.selectedMarker) return;
    elements.selectedMarker.textContent = current.selectedName ? current.selectedName : "(none)";
  }

  function updateLabelVisibility() {
    if (!current.markerMap) return;
    const showAll = Boolean(elements.showLabels.checked);
    for (const [name, entry] of current.markerMap.entries()) {
      entry.label.visible = showAll || name === current.selectedName;
    }
  }

  function setSelectedMarker(name) {
    const next = name || null;
    if (current.selectedName === next) return;

    if (current.markerMap && current.selectedName) {
      const prev = current.markerMap.get(current.selectedName);
      if (prev) prev.material.emissive.setHex(0x000000);
    }

    current.selectedName = next;

    if (current.markerMap && current.selectedName) {
      const entry = current.markerMap.get(current.selectedName);
      if (entry) entry.material.emissive.setHex(0x66c2ff);
    }

    updateSelectedUi();
    updateLabelVisibility();
  }

  function tick() {
    requestAnimationFrame(tick);
    viz.renderer.render(viz.scene, viz.camera);
    viz.labelRenderer.render(viz.scene, viz.camera);
  }
  requestAnimationFrame(tick);

  function clearScene() {
    viz.modelGroup.clear();
    viz.overlayGroup.clear();
    viz.manualGroup.clear();
    viz.smplGroup.clear();
    viz.overlayGroup.add(viz.manualGroup);
    viz.overlayGroup.add(viz.smplGroup);
    if (previewMesh) {
      previewMesh.removeFromParent();
      previewMesh = null;
    }
    elements.markerList.replaceChildren();
    current = {
      data: null,
      skeleton: null,
      bodies: null,
      markers: null,
      markerMap: null,
      selectedName: null,
      manualSelecting: false,
      pendingPick: null,
      picks: [],
      pickId: 1,
      smpl: null,
    };
    updateSelectedUi();
    updatePickUi();
  }

  async function rebuildModelMesh() {
    viz.modelGroup.clear();
    viz.modelGroup.visible = Boolean(elements.showMesh?.checked);
    if (!viz.modelGroup.visible) return;
    if (!current.data) return;

    const bodyEntries = Object.entries(current.data.bodies || {}).filter(([, e]) => e?.translation);
    const geometryFiles = new Set();
    for (const [, entry] of bodyEntries) {
      for (const f of entry.attachedGeometries || []) geometryFiles.add(f);
    }

    const missing = [];
    await Promise.all(
      [...geometryFiles].map(async (f) => {
        try {
          await loadVtpGeometry(f);
        } catch {
          missing.push(f);
        }
      }),
    );

    for (let i = 0; i < bodyEntries.length; i++) {
      const [name, body] = bodyEntries[i];
      const attached = body.attachedGeometries || [];
      if (!attached.length) continue;

      const group = new THREE.Group();
      group.name = `body:${name}`;

      const t = body.translation || [0, 0, 0];
      group.position.set(t[0], t[1], t[2]);

      const r = body.rotation || [0, 0, 0];
      group.rotation.set(r[0], r[1], r[2], "XYZ");

      const s = body.scaleFactors || [1, 1, 1];
      group.scale.set(s[0], s[1], s[2]);

      const color = new THREE.Color();
      color.setHSL((i / Math.max(1, bodyEntries.length)) * 0.9, 0.45, 0.55);
      const material = new THREE.MeshStandardMaterial({ color, roughness: 0.85, metalness: 0.05 });

      for (const geomFile of attached) {
        const geom = await loadVtpGeometry(geomFile).catch(() => null);
        if (!geom) continue;
        const mesh = new THREE.Mesh(geom, material);
        mesh.frustumCulled = false;
        group.add(mesh);
      }

      viz.modelGroup.add(group);
    }
    return missing;
  }

  async function loadAndShow() {
    const url = elements.scenePath.value.trim();
    if (!url) return;
    setMeta("Loading...");
    clearScene();

    const data = await loadSceneJson(url);
    current.data = data;

    const bodies = new Map();
    for (const [name, entry] of Object.entries(data.bodies || {})) {
      if (!entry?.translation) continue;
      bodies.set(name, entry.translation);
    }

    const edges = data.edges || [];
    const markers = data.markers || [];

    current.skeleton = buildSkeleton(viz.overlayGroup, bodies, edges);
    current.bodies = buildBodyPoints(viz.overlayGroup, bodies);

    const markerRadius = Number(elements.markerSize.value || "0.02");
    current.markers = buildMarkers(viz.overlayGroup, markers, markerRadius, elements.showLabels.checked);
    current.markerMap = current.markers.markerMap;

    setSelectedMarker(null);
    renderMarkerList(current.markerMap);
    applyMarkerFilter();

    // Toggle initial visibility.
    current.skeleton.group.visible = Boolean(elements.showSkeleton.checked);
    current.bodies.group.visible = Boolean(elements.showBodies.checked);
    current.markers.group.visible = Boolean(elements.showMarkers.checked);
    viz.modelGroup.visible = Boolean(elements.showMesh?.checked);
    viz.grid.visible = Boolean(elements.showGrid.checked);
    viz.axes.visible = Boolean(elements.showAxes.checked);

    // Camera framing.
    const points = [];
    for (const p of bodies.values()) points.push(p);
    for (const m of markers) points.push(m.position);
    if (points.length) {
      const box = computeBounds(points);
      fitCameraToBox(viz.camera, viz.controls, box);
    }

    const modelPath = data?.meta?.model_path || "unknown";
    let meshCount = 0;
    for (const body of Object.values(data.bodies || {})) meshCount += (body?.attachedGeometries || []).length;
    const baseMeta = `model: ${modelPath}\nmarkers: ${markers.length}  bodies: ${bodies.size}  edges: ${edges.length}  meshes: ${meshCount}`;
    setMeta(baseMeta);

    if (viz.modelGroup.visible) {
      setMeta(`${baseMeta}\nLoading mesh... (first time may be slower)`);
      const missing = (await rebuildModelMesh()) || [];
      if (missing.length) {
        const preview = missing.slice(0, 6).map((s) => String(s)).join(", ");
        setMeta(`${baseMeta}\nmesh: Loaded (missing ${missing.length} VTP files, e.g., ${preview})`);
      } else {
        setMeta(`${baseMeta}\nmesh: Loaded`);
      }
    }

    // Load SMPL mesh if provided
    await ensureSmplMesh(viz, current, baseMeta);
  }

  elements.load.addEventListener("click", () => loadAndShow().catch((e) => setMeta(String(e))));
  elements.markerFilter.addEventListener("input", applyMarkerFilter);
  elements.showAllMarkers?.addEventListener("click", () => {
    if (!current.markerMap) return;
    for (const [name, entry] of current.markerMap.entries()) {
      entry.mesh.visible = true;
      const item = elements.markerList.querySelector(`.marker-item[data-name="${name}"] input[type="checkbox"]`);
      if (item) item.checked = true;
    }
  });
  elements.hideAllMarkers?.addEventListener("click", () => {
    if (!current.markerMap) return;
    for (const [name, entry] of current.markerMap.entries()) {
      entry.mesh.visible = false;
      const item = elements.markerList.querySelector(`.marker-item[data-name="${name}"] input[type="checkbox"]`);
      if (item) item.checked = false;
    }
  });

  function pickMarkerFromEvent(event) {
    if (!current.markers?.group) return;
    if (!current.markers.group.visible) return;

    const rect = viz.renderer.domElement.getBoundingClientRect();
    if (!rect.width || !rect.height) return;

    pointer.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
    pointer.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;

    raycaster.setFromCamera(pointer, viz.camera);
    const hits = raycaster.intersectObjects(current.markers.group.children, true);
    if (!hits.length) {
      setSelectedMarker(null);
      return;
    }

    let obj = hits[0].object;
    while (obj && obj.parent && !obj.userData?.markerName && !String(obj.name || "").startsWith("marker:")) obj = obj.parent;
    const markerName = obj?.userData?.markerName || (String(obj?.name || "").startsWith("marker:") ? obj.name.slice("marker:".length) : null);
    if (markerName) setSelectedMarker(markerName);
  }

  function pickManual(event) {
    if (!current.data) return;
    const rect = viz.renderer.domElement.getBoundingClientRect();
    if (!rect.width || !rect.height) return;
    pointer.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
    pointer.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;
    raycaster.setFromCamera(pointer, viz.camera);

    // 1) markers
    if (current.markers?.group?.visible) {
      const hits = raycaster.intersectObjects(current.markers.group.children, true);
      if (hits.length) {
        let obj = hits[0].object;
        while (obj && obj.parent && !obj.userData?.markerName && !String(obj.name || "").startsWith("marker:"))
          obj = obj.parent;
        const markerName =
          obj?.userData?.markerName || (String(obj?.name || "").startsWith("marker:") ? obj.name.slice("marker:".length) : null);
        if (markerName) {
          const pos = obj.getWorldPosition(new THREE.Vector3());
          current.pendingPick = { type: "marker", name: markerName, position: [pos.x, pos.y, pos.z] };
          showPreview(current.pendingPick.position);
          updatePickUi();
          return;
        }
      }
    }

    // 2) mesh vertex
    const meshes = [];
    if (viz.smplGroup.visible) {
      viz.smplGroup.traverse((o) => {
        if (o.isMesh) meshes.push(o);
      });
    }
    viz.modelGroup.traverse((o) => {
      if (o.isMesh) meshes.push(o);
    });
    if (meshes.length) {
      const hits = raycaster.intersectObjects(meshes, true);
      if (hits.length) {
        const hit = hits[0];
        const geom = hit.object.geometry;
        const posAttr = geom?.attributes?.position;
        if (posAttr) {
          const face = hit.face;
          const ids = face ? [face.a, face.b, face.c] : [0, 1, 2];
          let best = null;
          let bestDist = Infinity;
          const worldPoint = hit.point.clone();
          for (const vi of ids) {
            const v = new THREE.Vector3(posAttr.getX(vi), posAttr.getY(vi), posAttr.getZ(vi));
            hit.object.localToWorld(v);
            const d = v.distanceTo(worldPoint);
            if (d < bestDist) {
              bestDist = d;
              best = { vertex: vi, position: [v.x, v.y, v.z] };
            }
          }
          if (best) {
            const meshType = hit.object.userData?.meshType || "mesh";
            const meshPath = meshType === "smpl" ? current.smpl?.path || null : null;
            current.pendingPick = {
              type: "mesh",
              geom: hit.object.name || "(mesh)",
              vertex: best.vertex,
              position: best.position,
              meshType,
              meshPath,
            };
            showPreview(current.pendingPick.position);
            updatePickUi();
            return;
          }
        }
      }
    }
  }

  // Use capture phase so OrbitControls can't swallow events.
  viz.renderer.domElement.addEventListener(
    "pointerdown",
    (event) => {
      clickState.down = { x: event.clientX, y: event.clientY, id: event.pointerId };
    },
    { capture: true },
  );
  viz.renderer.domElement.addEventListener(
    "pointerup",
    (event) => {
      const down = clickState.down;
      clickState.down = null;
      if (!down || down.id !== event.pointerId) return;
      const dx = event.clientX - down.x;
      const dy = event.clientY - down.y;
      if (dx * dx + dy * dy > 25) return; // treat as drag if moved >5px
      if (current.manualSelecting) {
        pickManual(event);
      } else {
        pickMarkerFromEvent(event);
      }
    },
    { capture: true },
  );

  elements.showMesh?.addEventListener("change", () => {
    viz.modelGroup.visible = Boolean(elements.showMesh?.checked);
    if (viz.modelGroup.visible && current.data && viz.modelGroup.children.length === 0) {
      rebuildModelMesh().catch((e) => setMeta(String(e)));
    }
  });
  elements.showSkeleton.addEventListener("change", () => {
    if (!current.skeleton) return;
    current.skeleton.group.visible = Boolean(elements.showSkeleton.checked);
  });
  elements.showBodies.addEventListener("change", () => {
    if (!current.bodies) return;
    current.bodies.group.visible = Boolean(elements.showBodies.checked);
  });
  elements.showMarkers.addEventListener("change", () => {
    if (!current.markers) return;
    current.markers.group.visible = Boolean(elements.showMarkers.checked);
  });
  elements.showSmpl?.addEventListener("change", () => {
    viz.smplGroup.visible = Boolean(elements.showSmpl?.checked);
    if (viz.smplGroup.visible && !viz.smplGroup.children.length) {
      const baseMeta = elements.meta.textContent || "";
      ensureSmplMesh(viz, current, baseMeta).catch((e) => setMeta(String(e)));
    }
  });
  elements.showGrid.addEventListener("change", () => {
    viz.grid.visible = Boolean(elements.showGrid.checked);
  });
  elements.showAxes.addEventListener("change", () => {
    viz.axes.visible = Boolean(elements.showAxes.checked);
  });
  elements.showLabels.addEventListener("change", () => {
    updateLabelVisibility();
  });
  elements.markerSize.addEventListener("input", () => {
    if (!current.markerMap) return;
    const r = Number(elements.markerSize.value || "0.02");
    for (const entry of current.markerMap.values()) {
      entry.mesh.scale.setScalar(r);
      entry.label.position.set(0, r * 1.2, 0);
    }
  });

  function savePendingPick() {
    if (!current.pendingPick) return;
    const note = (elements.pickNote?.value || "").trim() || null;
    const rec = {
      id: current.pickId++,
      scene: elements.scenePath.value.trim(),
      type: current.pendingPick.type,
      name: current.pendingPick.name || null,
      geom: current.pendingPick.geom || null,
      vertex: current.pendingPick.vertex ?? null,
      position: current.pendingPick.position,
      meshType: current.pendingPick.meshType || null,
      meshPath: current.pendingPick.meshPath || null,
      note,
      timestamp: new Date().toISOString(),
    };
    current.picks.push(rec);

    // Visual marker
    const g = new THREE.SphereGeometry(1, 10, 8);
    const mat = new THREE.MeshStandardMaterial({ color: 0x00ff99, emissive: 0x004422, roughness: 0.45, metalness: 0.1 });
    const mesh = new THREE.Mesh(g, mat);
    const savedR = Number(elements.savedSize?.value || "0.015");
    mesh.scale.setScalar(savedR);
    mesh.position.set(rec.position[0], rec.position[1], rec.position[2]);
    mesh.name = `manual:${rec.id}`;
    viz.manualGroup.add(mesh);

    current.pendingPick = null;
    current.manualSelecting = false;
    if (elements.pickNote) elements.pickNote.value = "";
    if (previewMesh) {
      previewMesh.removeFromParent();
      previewMesh = null;
    }
    updatePickUi();
  }

  function exportPicks() {
    if (!current.picks.length) return;
    const blob = new Blob([JSON.stringify({ picks: current.picks }, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "manual_picks.json";
    a.click();
    URL.revokeObjectURL(url);
  }

  elements.startPick?.addEventListener("click", () => {
    current.manualSelecting = true;
    current.pendingPick = null;
    updatePickUi();
  });
  elements.savePick?.addEventListener("click", () => savePendingPick());
  elements.cancelPick?.addEventListener("click", () => {
    current.pendingPick = null;
    current.manualSelecting = false;
    if (elements.pickNote) elements.pickNote.value = "";
    if (previewMesh) {
      previewMesh.removeFromParent();
      previewMesh = null;
    }
    updatePickUi();
  });
  elements.exportPicks?.addEventListener("click", exportPicks);
  elements.previewSize?.addEventListener("input", () => {
    if (previewMesh) {
      const r = Number(elements.previewSize.value || "0.015");
      previewMesh.scale.setScalar(r);
    }
  });
  elements.savedSize?.addEventListener("input", () => {
    // Only affects future saved points; preview size remains separate.
  });

  await loadAndShow().catch((e) => setMeta(String(e)));
}

start().catch((e) => setMeta(String(e)));
