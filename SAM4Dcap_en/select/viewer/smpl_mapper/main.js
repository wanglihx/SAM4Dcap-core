import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { CSS2DObject, CSS2DRenderer } from "three/addons/renderers/CSS2DRenderer.js";

const elements = {
  opencapScenePath: document.getElementById("opencapScenePath"),
  smplMeshPath: document.getElementById("smplMeshPath"),
  load: document.getElementById("load"),
  exportJson: document.getElementById("exportJson"),
  exportCsv: document.getElementById("exportCsv"),
  importJson: document.getElementById("importJson"),
  meta: document.getElementById("meta"),

  layoutMode: document.getElementById("layoutMode"),
  splitDistance: document.getElementById("splitDistance"),
  smplYawPreset: document.getElementById("smplYawPreset"),
  lockScaleToBsm: document.getElementById("lockScaleToBsm"),
  bboxAlign: document.getElementById("bboxAlign"),
  fitAlign: document.getElementById("fitAlign"),
  resetSmpl: document.getElementById("resetSmpl"),
  fitCamera: document.getElementById("fitCamera"),

  showOpenCapMesh: document.getElementById("showOpenCapMesh"),
  showOpenCapSkeleton: document.getElementById("showOpenCapSkeleton"),
  showOpenCapMarkers: document.getElementById("showOpenCapMarkers"),
  showOpenCapLabels: document.getElementById("showOpenCapLabels"),
  showSmplMesh: document.getElementById("showSmplMesh"),
  showSmplWireframe: document.getElementById("showSmplWireframe"),
  showSmplMarkers: document.getElementById("showSmplMarkers"),
  showMappedPoints: document.getElementById("showMappedPoints"),
  showGrid: document.getElementById("showGrid"),
  showAxes: document.getElementById("showAxes"),

  markerSize: document.getElementById("markerSize"),
  markerFilter: document.getElementById("markerFilter"),
  markerList: document.getElementById("markerList"),

  selectedMarker: document.getElementById("selectedMarker"),
  selectedSmpl: document.getElementById("selectedSmpl"),
  clearCurrent: document.getElementById("clearCurrent"),
  clearAll: document.getElementById("clearAll"),
  autoAdvance: document.getElementById("autoAdvance"),

  viewer: document.getElementById("viewer"),
};

const geometryCache = new Map();
const GEOMETRY_BASE = "../Geometry";

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

async function loadJson(url, hint) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`${hint} fetch failed: ${res.status} ${res.statusText} (${url})`);
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
    const url = `${GEOMETRY_BASE}/${encodeURIComponent(base)}`;
    const res = await fetch(url);
    if (!res.ok) throw new Error(`Failed to fetch VTP: ${res.status} ${res.statusText} (${url})`);
    const text = await res.text();
    return parseVtpAscii(text);
  })();

  geometryCache.set(base, promise);
  return promise;
}

function computeBounds(points) {
  const box = new THREE.Box3();
  for (const p of points) box.expandByPoint(p);
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

  const openCapLayout = new THREE.Group();
  openCapLayout.name = "opencapLayout";
  const openCap = new THREE.Group();
  openCap.name = "opencap";
  openCapLayout.add(openCap);
  root.add(openCapLayout);

  const smplLayout = new THREE.Group();
  smplLayout.name = "smplLayout";
  const smpl = new THREE.Group();
  smpl.name = "smplAligned";
  smplLayout.add(smpl);
  root.add(smplLayout);

  window.addEventListener("resize", () => resize(renderer, labelRenderer, camera, container));
  resize(renderer, labelRenderer, camera, container);

  return { scene, camera, renderer, labelRenderer, controls, grid, axes, root, openCapLayout, openCap, smplLayout, smpl };
}

function markerColor(name) {
  const n = name.toLowerCase();
  if (n.includes("asis") || n.includes("psis") || n.includes("hjc") || n.includes("c7")) return 0x8bd3ff;
  if (n.includes("thigh") || n.includes("knee") || n.includes("ankle") || n.includes("toe") || n.includes("calc") || n.includes("meta"))
    return 0x7cffa7;
  if (n.includes("shoulder") || n.includes("elbow") || n.includes("wrist") || n.includes("sh")) return 0xffc27a;
  return 0xd6d6d6;
}

function buildOpenCapSkeleton(root, bodies, edges) {
  const group = new THREE.Group();
  group.name = "opencapSkeleton";

  const validEdges = (edges || []).filter(([a, b]) => bodies.has(a) && bodies.has(b));

  const positions = new Float32Array(validEdges.length * 2 * 3);
  for (let i = 0; i < validEdges.length; i++) {
    const [a, b] = validEdges[i];
    const pa = bodies.get(a);
    const pb = bodies.get(b);
    const base = i * 2 * 3;
    positions[base + 0] = pa.x;
    positions[base + 1] = pa.y;
    positions[base + 2] = pa.z;
    positions[base + 3] = pb.x;
    positions[base + 4] = pb.y;
    positions[base + 5] = pb.z;
  }

  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));
  const material = new THREE.LineBasicMaterial({ color: 0x66c2ff });
  const lines = new THREE.LineSegments(geometry, material);
  group.add(lines);

  root.add(group);
  return { group };
}

function buildOpenCapMarkers(root, markers, markerRadius, showLabels) {
  const group = new THREE.Group();
  group.name = "opencapMarkers";

  const sphere = new THREE.SphereGeometry(1, 12, 10);
  const markerMap = new Map();

  for (const { name, position } of markers || []) {
    const material = new THREE.MeshStandardMaterial({
      color: 0x3399ff, // blue for OpenCap
      roughness: 0.65,
      metalness: 0.05,
      emissive: 0x000000,
    });
    const mesh = new THREE.Mesh(sphere, material);
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
    markerMap.set(name, { mesh, label, material, position: new THREE.Vector3(position[0], position[1], position[2]) });
  }

  root.add(group);
  return { group, markerMap };
}

async function loadSmplMarkers(root, markerRadius) {
  const url = "smpl_markers_aligned.json";
  let data = null;
  try {
    const res = await fetch(url);
    if (!res.ok) return;
    data = await res.json();
  } catch {
    return;
  }
  if (!data?.markers?.length) return;
  if (current.smpl.smplMarkers) {
    root.remove(current.smpl.smplMarkers.group);
    current.smpl.smplMarkers = null;
  }
  const group = new THREE.Group();
  group.name = "smplMarkersAligned";
  const sphere = new THREE.SphereGeometry(1, 12, 10);
  for (const { name, position } of data.markers) {
    const material = new THREE.MeshStandardMaterial({
      color: 0xff3333, // red for SMPL
      roughness: 0.65,
      metalness: 0.05,
      emissive: 0x000000,
    });
    const mesh = new THREE.Mesh(sphere, material);
    mesh.position.set(position[0], position[1], position[2]);
    mesh.scale.setScalar(markerRadius);
    mesh.name = `smpl_marker:${name}`;
    group.add(mesh);
  }
  root.add(group);
  current.smpl.smplMarkers = { group };
}

async function buildOpenCapMesh(root, sceneData, missingOut) {
  const group = new THREE.Group();
  group.name = "opencapMesh";

  const bodyEntries = Object.entries(sceneData?.bodies || {}).filter(([, e]) => e?.translation);
  const geometryFiles = new Set();
  for (const [, entry] of bodyEntries) {
    for (const f of entry.attachedGeometries || []) geometryFiles.add(f);
  }

  await Promise.all(
    [...geometryFiles].map(async (f) => {
      try {
        await loadVtpGeometry(f);
      } catch {
        missingOut.push(f);
      }
    }),
  );

  for (let i = 0; i < bodyEntries.length; i++) {
    const [name, body] = bodyEntries[i];
    const attached = body.attachedGeometries || [];
    if (!attached.length) continue;

    const bodyGroup = new THREE.Group();
    bodyGroup.name = `body:${name}`;

    const t = body.translation || [0, 0, 0];
    bodyGroup.position.set(t[0], t[1], t[2]);

    const r = body.rotation || [0, 0, 0];
    bodyGroup.rotation.set(r[0], r[1], r[2], "XYZ");

    const s = body.scaleFactors || [1, 1, 1];
    bodyGroup.scale.set(s[0], s[1], s[2]);

    const color = new THREE.Color();
    color.setHSL((i / Math.max(1, bodyEntries.length)) * 0.9, 0.45, 0.55);
    const material = new THREE.MeshStandardMaterial({ color, roughness: 0.85, metalness: 0.05 });

    for (const geomFile of attached) {
      const geom = await loadVtpGeometry(geomFile).catch(() => null);
      if (!geom) continue;
      const mesh = new THREE.Mesh(geom, material);
      mesh.frustumCulled = false;
      bodyGroup.add(mesh);
    }

    group.add(bodyGroup);
  }

  root.add(group);
  return { group };
}

function buildSmplMesh(root, smplMeshData) {
  const positions = Float32Array.from(smplMeshData.vertices || []);
  const facesRaw = smplMeshData.faces || smplMeshData.triangles || [];
  const faces = Uint32Array.from(facesRaw);
  const index = positions.length / 3 > 65535 ? new Uint32Array(faces) : new Uint16Array(faces);

  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));
  geometry.setIndex(new THREE.BufferAttribute(index, 1));
  geometry.computeVertexNormals();
  geometry.computeBoundingSphere();

  const material = new THREE.MeshStandardMaterial({
    color: 0xd8c1a3,
    roughness: 0.85,
    metalness: 0.02,
    transparent: true,
    opacity: 0.92,
  });

  const mesh = new THREE.Mesh(geometry, material);
  mesh.name = "smplMesh";

  root.add(mesh);
  return { mesh, geometry, material, positions, index };
}

function barycentricCoords(a, b, c, p) {
  const v0 = b.clone().sub(a);
  const v1 = c.clone().sub(a);
  const v2 = p.clone().sub(a);
  const d00 = v0.dot(v0);
  const d01 = v0.dot(v1);
  const d11 = v1.dot(v1);
  const d20 = v2.dot(v0);
  const d21 = v2.dot(v1);
  const denom = d00 * d11 - d01 * d01;
  if (Math.abs(denom) < 1e-12) return [1, 0, 0];
  const v = (d11 * d20 - d01 * d21) / denom;
  const w = (d00 * d21 - d01 * d20) / denom;
  const u = 1 - v - w;
  return [u, v, w];
}

function downloadText(filename, text, mime = "application/json") {
  const blob = new Blob([text], { type: mime });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
}

function fmtVec3(v) {
  return `${v.x.toFixed(4)},${v.y.toFixed(4)},${v.z.toFixed(4)}`;
}

function jacobiEigenSymmetric3(A) {
  // A: [[a00,a01,a02],[a01,a11,a12],[a02,a12,a22]]
  const D = [
    [A[0][0], A[0][1], A[0][2]],
    [A[1][0], A[1][1], A[1][2]],
    [A[2][0], A[2][1], A[2][2]],
  ];
  const V = [
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1],
  ];
  const maxIter = 32;
  const eps = 1e-12;

  function rotate(p, q) {
    const app = D[p][p];
    const aqq = D[q][q];
    const apq = D[p][q];
    if (Math.abs(apq) < eps) return;

    const phi = 0.5 * Math.atan2(2 * apq, aqq - app);
    const c = Math.cos(phi);
    const s = Math.sin(phi);

    for (let i = 0; i < 3; i++) {
      if (i === p || i === q) continue;
      const dip = D[i][p];
      const diq = D[i][q];
      D[i][p] = c * dip - s * diq;
      D[p][i] = D[i][p];
      D[i][q] = s * dip + c * diq;
      D[q][i] = D[i][q];
    }

    const dpp = c * c * app - 2 * s * c * apq + s * s * aqq;
    const dqq = s * s * app + 2 * s * c * apq + c * c * aqq;
    D[p][p] = dpp;
    D[q][q] = dqq;
    D[p][q] = 0;
    D[q][p] = 0;

    for (let i = 0; i < 3; i++) {
      const vip = V[i][p];
      const viq = V[i][q];
      V[i][p] = c * vip - s * viq;
      V[i][q] = s * vip + c * viq;
    }
  }

  for (let iter = 0; iter < maxIter; iter++) {
    let p = 0;
    let q = 1;
    let max = Math.abs(D[0][1]);
    const a02 = Math.abs(D[0][2]);
    const a12 = Math.abs(D[1][2]);
    if (a02 > max) {
      max = a02;
      p = 0;
      q = 2;
    }
    if (a12 > max) {
      max = a12;
      p = 1;
      q = 2;
    }
    if (max < 1e-10) break;
    rotate(p, q);
  }

  const eigenvalues = [D[0][0], D[1][1], D[2][2]];
  const eigenvectors = [
    [V[0][0], V[1][0], V[2][0]],
    [V[0][1], V[1][1], V[2][1]],
    [V[0][2], V[1][2], V[2][2]],
  ];

  // sort descending by eigenvalue
  const idx = [0, 1, 2].sort((i, j) => eigenvalues[j] - eigenvalues[i]);
  return {
    eigenvalues: idx.map((i) => eigenvalues[i]),
    eigenvectors: idx.map((i) => eigenvectors[i]),
  };
}

function det3(m) {
  return (
    m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1]) -
    m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0]) +
    m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
  );
}

function mul3(A, B) {
  const out = [
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0],
  ];
  for (let i = 0; i < 3; i++) {
    for (let j = 0; j < 3; j++) {
      out[i][j] = A[i][0] * B[0][j] + A[i][1] * B[1][j] + A[i][2] * B[2][j];
    }
  }
  return out;
}

function transpose3(A) {
  return [
    [A[0][0], A[1][0], A[2][0]],
    [A[0][1], A[1][1], A[2][1]],
    [A[0][2], A[1][2], A[2][2]],
  ];
}

function matVec3(A, v) {
  return [
    A[0][0] * v[0] + A[0][1] * v[1] + A[0][2] * v[2],
    A[1][0] * v[0] + A[1][1] * v[1] + A[1][2] * v[2],
    A[2][0] * v[0] + A[2][1] * v[1] + A[2][2] * v[2],
  ];
}

function normalize3(v) {
  const n = Math.hypot(v[0], v[1], v[2]) || 1;
  return [v[0] / n, v[1] / n, v[2] / n];
}

function dot3(a, b) {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

function cross3(a, b) {
  return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
}

function orthonormalizeColumns3(M) {
  // M: columns as vectors [[c0],[c1],[c2]] each len3
  const c0 = normalize3(M[0]);
  let c1 = [
    M[1][0] - dot3(M[1], c0) * c0[0],
    M[1][1] - dot3(M[1], c0) * c0[1],
    M[1][2] - dot3(M[1], c0) * c0[2],
  ];
  c1 = normalize3(c1);
  let c2 = [
    M[2][0] - dot3(M[2], c0) * c0[0] - dot3(M[2], c1) * c1[0],
    M[2][1] - dot3(M[2], c0) * c0[1] - dot3(M[2], c1) * c1[1],
    M[2][2] - dot3(M[2], c0) * c0[2] - dot3(M[2], c1) * c1[2],
  ];
  const n2 = Math.hypot(c2[0], c2[1], c2[2]);
  if (n2 < 1e-8) {
    c2 = normalize3(cross3(c0, c1));
  } else {
    c2 = [c2[0] / n2, c2[1] / n2, c2[2] / n2];
  }
  return [c0, c1, c2];
}

function svd3x3(M) {
  // M: 3x3
  const Mt = transpose3(M);
  const A = mul3(Mt, M); // symmetric
  const { eigenvalues, eigenvectors } = jacobiEigenSymmetric3(A);

  const singularValues = eigenvalues.map((l) => Math.sqrt(Math.max(0, l)));
  const V = [
    [eigenvectors[0][0], eigenvectors[1][0], eigenvectors[2][0]],
    [eigenvectors[0][1], eigenvectors[1][1], eigenvectors[2][1]],
    [eigenvectors[0][2], eigenvectors[1][2], eigenvectors[2][2]],
  ];

  const Ucols = [];
  for (let i = 0; i < 3; i++) {
    const sigma = singularValues[i];
    const vi = [V[0][i], V[1][i], V[2][i]];
    const mvi = matVec3(M, vi);
    if (sigma > 1e-10) {
      Ucols.push([mvi[0] / sigma, mvi[1] / sigma, mvi[2] / sigma]);
    } else {
      Ucols.push([0, 0, 0]);
    }
  }

  const UcolsOrtho = orthonormalizeColumns3(Ucols);
  const U = [
    [UcolsOrtho[0][0], UcolsOrtho[1][0], UcolsOrtho[2][0]],
    [UcolsOrtho[0][1], UcolsOrtho[1][1], UcolsOrtho[2][1]],
    [UcolsOrtho[0][2], UcolsOrtho[1][2], UcolsOrtho[2][2]],
  ];

  return { U, S: singularValues, V };
}

function fitSimilarityTransform(sourcePoints, targetPoints) {
  // Finds s,R,t s.t. y ~= s R x + t
  const n = Math.min(sourcePoints.length, targetPoints.length);
  if (n < 3) throw new Error("Alignment requires at least 3 mapped points");

  const muX = new THREE.Vector3();
  const muY = new THREE.Vector3();
  for (let i = 0; i < n; i++) {
    muX.add(sourcePoints[i]);
    muY.add(targetPoints[i]);
  }
  muX.multiplyScalar(1 / n);
  muY.multiplyScalar(1 / n);

  let varX = 0;
  const cov = [
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0],
  ];
  for (let i = 0; i < n; i++) {
    const x = sourcePoints[i].clone().sub(muX);
    const y = targetPoints[i].clone().sub(muY);
    varX += x.lengthSq();
    cov[0][0] += y.x * x.x;
    cov[0][1] += y.x * x.y;
    cov[0][2] += y.x * x.z;
    cov[1][0] += y.y * x.x;
    cov[1][1] += y.y * x.y;
    cov[1][2] += y.y * x.z;
    cov[2][0] += y.z * x.x;
    cov[2][1] += y.z * x.y;
    cov[2][2] += y.z * x.z;
  }

  varX /= n;
  for (let r = 0; r < 3; r++) for (let c = 0; c < 3; c++) cov[r][c] /= n;
  if (varX < 1e-12) throw new Error("Alignment failed: source variance too small");

  const { U, S, V } = svd3x3(cov);
  const Vt = transpose3(V);
  let R = mul3(U, Vt);

  const detR = det3(R);
  const Sfix = [
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, detR < 0 ? -1 : 1],
  ];
  if (detR < 0) {
    R = mul3(mul3(U, Sfix), Vt);
  }

  const traceDS = S[0] * Sfix[0][0] + S[1] * Sfix[1][1] + S[2] * Sfix[2][2];
  const scale = traceDS / varX;

  const muXarr = [muX.x, muX.y, muX.z];
  const rmuX = matVec3(R, muXarr);
  const t = new THREE.Vector3(muY.x - scale * rmuX[0], muY.y - scale * rmuX[1], muY.z - scale * rmuX[2]);

  return { scale, R, t };
}

function applyLayout(viz) {
  const mode = String(elements.layoutMode.value || "overlay");
  const d = Number(elements.splitDistance.value || "1.2");
  if (mode === "split") {
    viz.openCapLayout.position.set(-d / 2, 0, 0);
    viz.smplLayout.position.set(d / 2, 0, 0);
  } else {
    viz.openCapLayout.position.set(0, 0, 0);
    viz.smplLayout.position.set(0, 0, 0);
  }
}

function start() {
  const viz = buildScene(elements.viewer);
  const raycaster = new THREE.Raycaster();
  const pointer = new THREE.Vector2();
  const clickState = { down: null };

  let current = {
    opencap: { data: null, skeleton: null, markers: null, markerMap: null, mesh: null, selectedMarker: null },
    smpl: { data: null, mesh: null, geom: null, material: null, positions: null, index: null, pick: null, mappedGroup: null },
    mapping: {},
    selectedSmpl: null,
  };

  function updateSelectedMarkerUi() {
    elements.selectedMarker.textContent = current.opencap.selectedMarker ? current.opencap.selectedMarker : "(none)";
  }

  function updateSelectedSmplUi() {
    if (!current.selectedSmpl) {
      elements.selectedSmpl.textContent = "(none)";
      return;
    }
    const s = current.selectedSmpl;
    elements.selectedSmpl.textContent = `v${s.vertex} f(${s.face.join(",")}) bary(${s.barycentric.map((x) => x.toFixed(3)).join(",")}) @ ${fmtVec3(s.point)}`;
  }

  function updateLabelVisibility() {
    if (!current.opencap.markerMap) return;
    const showAll = Boolean(elements.showOpenCapLabels.checked);
    for (const [name, entry] of current.opencap.markerMap.entries()) {
      entry.label.visible = showAll || name === current.opencap.selectedMarker;
    }
  }

  function setSelectedMarker(name) {
    const next = name || null;
    if (current.opencap.selectedMarker === next) return;

    if (current.opencap.markerMap && current.opencap.selectedMarker) {
      const prev = current.opencap.markerMap.get(current.opencap.selectedMarker);
      if (prev) prev.material.emissive.setHex(0x000000);
    }
    current.opencap.selectedMarker = next;
    if (current.opencap.markerMap && next) {
      const entry = current.opencap.markerMap.get(next);
      if (entry) entry.material.emissive.setHex(0x66c2ff);
    }
    updateSelectedMarkerUi();
    updateLabelVisibility();
    renderMarkerList();
  }

  function clearAll() {
    current.mapping = {};
    rebuildMappedPoints();
    renderMarkerList();
  }

  function clearCurrent() {
    const name = current.opencap.selectedMarker;
    if (!name) return;
    delete current.mapping[name];
    rebuildMappedPoints();
    renderMarkerList();
  }

  function markerNames() {
    if (!current.opencap.markerMap) return [];
    return [...current.opencap.markerMap.keys()].sort();
  }

  function renderMarkerList() {
    elements.markerList.replaceChildren();
    const names = markerNames();
    for (const name of names) {
      const item = document.createElement("div");
      item.className = "marker-item";
      item.dataset.name = name;

      const checkbox = document.createElement("input");
      checkbox.type = "checkbox";
      checkbox.checked = true;
      checkbox.addEventListener("change", (e) => {
        const entry = current.opencap.markerMap?.get(name);
        if (!entry) return;
        entry.mesh.visible = checkbox.checked;
        e.stopPropagation();
      });

      const code = document.createElement("code");
      code.textContent = name;

      const map = document.createElement("span");
      const m = current.mapping[name];
      if (m) {
        map.className = "map";
        map.textContent = `v${m.vertex}`;
      } else {
        map.className = "map missing";
        map.textContent = "—";
      }

      if (name === current.opencap.selectedMarker) {
        item.style.outline = "1px solid rgba(102,194,255,0.55)";
        item.style.background = "rgba(102,194,255,0.08)";
      }

      item.appendChild(checkbox);
      item.appendChild(code);
      item.appendChild(map);
      item.addEventListener("click", () => setSelectedMarker(name));

      elements.markerList.appendChild(item);
    }
    applyMarkerFilter();
  }

  function applyMarkerFilter() {
    const q = elements.markerFilter.value.trim().toLowerCase();
    const items = elements.markerList.querySelectorAll(".marker-item");
    for (const item of items) {
      const name = (item.dataset.name || "").toLowerCase();
      item.style.display = !q || name.includes(q) ? "" : "none";
    }
  }

  function smplVertexPosition(i) {
    const p = current.smpl.positions;
    return new THREE.Vector3(p[i * 3 + 0], p[i * 3 + 1], p[i * 3 + 2]);
  }

  function findFirstFaceContainingVertex(vertexId) {
    const idx = current.smpl.index;
    if (!idx) return null;
    const v = Number(vertexId);
    for (let i = 0; i + 2 < idx.length; i += 3) {
      const a = idx[i];
      const b = idx[i + 1];
      const c = idx[i + 2];
      if (a === v || b === v || c === v) return [a, b, c];
    }
    return null;
  }

  function mappingFromVertex(vertexId) {
    const face = findFirstFaceContainingVertex(vertexId);
    if (!face) return null;
    const v = Number(vertexId);
    let bary = [0, 0, 0];
    if (face[0] === v) bary = [1, 0, 0];
    else if (face[1] === v) bary = [0, 1, 0];
    else bary = [0, 0, 1];
    return { vertex: v, face, barycentric: bary };
  }

  function smplBaryPointLocal(face, bary) {
    const a = smplVertexPosition(face[0]);
    const b = smplVertexPosition(face[1]);
    const c = smplVertexPosition(face[2]);
    return a.multiplyScalar(bary[0]).add(b.multiplyScalar(bary[1])).add(c.multiplyScalar(bary[2]));
  }

  function rebuildMappedPoints() {
    if (!current.smpl.mappedGroup) return;
    current.smpl.mappedGroup.clear();
    current.smpl.mappedGroup.visible = Boolean(elements.showMappedPoints.checked);
    if (!current.smpl.positions) return;

    const sphere = new THREE.SphereGeometry(1, 10, 8);
    const r = 0.012;
    const entries = Object.entries(current.mapping);
  for (const [markerName, m] of entries) {
    if (!m?.face || !m?.barycentric) continue;
    const local = smplBaryPointLocal(m.face, m.barycentric);
    const material = new THREE.MeshStandardMaterial({ color: 0xff3333, roughness: 0.55, metalness: 0.05 }); // red for SMPL mapped points
      const dot = new THREE.Mesh(sphere, material);
      dot.scale.setScalar(r);
      dot.position.copy(local);
      dot.name = `mapped:${markerName}`;
      current.smpl.mappedGroup.add(dot);
    }
  }

  function setSmplPick(pick) {
    current.selectedSmpl = pick;
    updateSelectedSmplUi();
    if (current.smpl.pick) {
      current.smpl.pick.visible = Boolean(pick);
      if (pick) current.smpl.pick.position.copy(pick.pointLocal);
    }
  }

  function nextUnmappedMarker(afterName) {
    const names = markerNames();
    if (!names.length) return null;
    const start = afterName ? Math.max(0, names.indexOf(afterName)) : 0;
    for (let i = start + 1; i < names.length; i++) if (!current.mapping[names[i]]) return names[i];
    for (let i = 0; i <= start; i++) if (!current.mapping[names[i]]) return names[i];
    return null;
  }

  function assignMappingForSelectedMarker(pick) {
    const markerName = current.opencap.selectedMarker;
    if (!markerName) return;
    current.mapping[markerName] = {
      vertex: pick.vertex,
      face: pick.face,
      barycentric: pick.barycentric,
    };
    rebuildMappedPoints();
    renderMarkerList();
    if (elements.autoAdvance.checked) {
      const next = nextUnmappedMarker(markerName);
      if (next) setSelectedMarker(next);
    }
  }

  function pickFromPointer(event) {
    const rect = viz.renderer.domElement.getBoundingClientRect();
    if (!rect.width || !rect.height) return { marker: null, smpl: null };
    pointer.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
    pointer.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;
    raycaster.setFromCamera(pointer, viz.camera);

    let markerName = null;
    if (current.opencap.markers?.group?.visible) {
      const hits = raycaster.intersectObjects(current.opencap.markers.group.children, true);
      if (hits.length) {
        let obj = hits[0].object;
        while (obj && obj.parent && !obj.userData?.markerName && !String(obj.name || "").startsWith("marker:")) obj = obj.parent;
        markerName =
          obj?.userData?.markerName || (String(obj?.name || "").startsWith("marker:") ? obj.name.slice("marker:".length) : null);
      }
    }

    let smplHit = null;
    if (!markerName && current.smpl.mesh && current.smpl.mesh.visible) {
      const hits = raycaster.intersectObject(current.smpl.mesh, true);
      if (hits.length) smplHit = hits[0];
    }

    return { marker: markerName, smpl: smplHit };
  }

  function buildPickFromSmplHit(hit) {
    if (!hit?.face || !current.smpl.index) return null;
    const { a, b, c } = hit.face;
    const face = [a, b, c];
    const localPoint = current.smpl.mesh.worldToLocal(hit.point.clone());
    const pa = smplVertexPosition(a);
    const pb = smplVertexPosition(b);
    const pc = smplVertexPosition(c);
    const bary = barycentricCoords(pa, pb, pc, localPoint);
    let vertex = a;
    if (bary[1] >= bary[0] && bary[1] >= bary[2]) vertex = b;
    if (bary[2] >= bary[0] && bary[2] >= bary[1]) vertex = c;
    return { vertex, face, barycentric: bary, point: hit.point.clone(), pointLocal: localPoint };
  }

  function tick() {
    requestAnimationFrame(tick);
    viz.renderer.render(viz.scene, viz.camera);
    viz.labelRenderer.render(viz.scene, viz.camera);
  }
  requestAnimationFrame(tick);

  function clearAllScene() {
    viz.openCap.clear();
    viz.smpl.clear();
    elements.markerList.replaceChildren();
    current = {
      opencap: { data: null, skeleton: null, markers: null, markerMap: null, mesh: null, selectedMarker: null },
      smpl: { data: null, mesh: null, geom: null, material: null, positions: null, index: null, pick: null, mappedGroup: null },
      mapping: {},
      selectedSmpl: null,
    };
    updateSelectedMarkerUi();
    updateSelectedSmplUi();
  }

  function setSmplYawPreset() {
    const deg = Number(elements.smplYawPreset.value || "0");
    viz.smpl.rotation.set(0, THREE.MathUtils.degToRad(deg), 0);
  }

  function getScaleToBsmFromMeta() {
    const s = Number(current.smpl.data?.meta?.scale_to_bsm_c7_ground);
    if (!Number.isFinite(s) || s <= 0) return null;
    return s;
  }

  function getLockedScaleOrNull() {
    if (!elements.lockScaleToBsm?.checked) return null;
    return getScaleToBsmFromMeta();
  }

  function resetSmplTransform() {
    viz.smpl.position.set(0, 0, 0);
    viz.smpl.rotation.set(0, 0, 0);
    const locked = getLockedScaleOrNull();
    if (locked) viz.smpl.scale.setScalar(locked);
    else viz.smpl.scale.set(1, 1, 1);
    setSmplYawPreset();
  }

  function roughAlignBBox() {
    if (!current.opencap.markerMap || !current.smpl.positions) return;
    const openPoints = [...current.opencap.markerMap.values()].map((e) => e.position.clone());
    const boxO = computeBounds(openPoints);

    const boxS = new THREE.Box3();
    const pos = current.smpl.positions;
    for (let i = 0; i < pos.length; i += 3) boxS.expandByPoint(new THREE.Vector3(pos[i], pos[i + 1], pos[i + 2]));

    const locked = getLockedScaleOrNull();
    const hO = boxO.max.y - boxO.min.y;
    const hS = boxS.max.y - boxS.min.y;
    const scale = locked ? locked : hS > 1e-6 ? hO / hS : 1;

    const centerO = boxO.getCenter(new THREE.Vector3());
    const centerS = boxS.getCenter(new THREE.Vector3());

    viz.smpl.scale.setScalar(scale);
    viz.smpl.position.copy(centerO.clone().sub(centerS.multiplyScalar(scale)));
  }

  function fitRigidTransformFixedScale(sourcePoints, targetPoints, fixedScale) {
    const n = Math.min(sourcePoints.length, targetPoints.length);
    if (n < 3) throw new Error("Alignment requires at least 3 mapped points");
    const scale = Number(fixedScale);
    if (!Number.isFinite(scale) || scale <= 0) throw new Error("Invalid fixed scale factor");

    const muX = new THREE.Vector3();
    const muY = new THREE.Vector3();
    for (let i = 0; i < n; i++) {
      muX.add(sourcePoints[i]);
      muY.add(targetPoints[i]);
    }
    muX.multiplyScalar(1 / n);
    muY.multiplyScalar(1 / n);

    const cov = [
      [0, 0, 0],
      [0, 0, 0],
      [0, 0, 0],
    ];
    for (let i = 0; i < n; i++) {
      const x = sourcePoints[i].clone().sub(muX);
      const y = targetPoints[i].clone().sub(muY);
      cov[0][0] += y.x * x.x;
      cov[0][1] += y.x * x.y;
      cov[0][2] += y.x * x.z;
      cov[1][0] += y.y * x.x;
      cov[1][1] += y.y * x.y;
      cov[1][2] += y.y * x.z;
      cov[2][0] += y.z * x.x;
      cov[2][1] += y.z * x.y;
      cov[2][2] += y.z * x.z;
    }
    for (let r = 0; r < 3; r++) for (let c = 0; c < 3; c++) cov[r][c] /= n;

    const { U, V } = svd3x3(cov);
    const Vt = transpose3(V);
    let R = mul3(U, Vt);

    const detR = det3(R);
    if (detR < 0) {
      const Sfix = [
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, -1],
      ];
      R = mul3(mul3(U, Sfix), Vt);
    }

    const muXarr = [muX.x, muX.y, muX.z];
    const rmuX = matVec3(R, muXarr);
    const t = new THREE.Vector3(muY.x - scale * rmuX[0], muY.y - scale * rmuX[1], muY.z - scale * rmuX[2]);
    return { scale, R, t };
  }

  function fitAlignFromMapping() {
    if (!current.opencap.markerMap || !current.smpl.positions) throw new Error("OpenCap/SMPL data not loaded yet");

    const src = [];
    const dst = [];
    for (const [markerName, m] of Object.entries(current.mapping)) {
      const marker = current.opencap.markerMap.get(markerName);
      if (!marker) continue;
      if (!m?.face || !m?.barycentric) continue;
      src.push(smplBaryPointLocal(m.face, m.barycentric));
      dst.push(marker.position.clone());
    }
    if (src.length < 3) throw new Error("Alignment requires at least 3 valid mappings (face + barycentric)");

    const { scale, R, t } = fitSimilarityTransform(src, dst);
    const m4 = new THREE.Matrix4();
    m4.set(R[0][0], R[0][1], R[0][2], 0, R[1][0], R[1][1], R[1][2], 0, R[2][0], R[2][1], R[2][2], 0, 0, 0, 0, 1);
    viz.smpl.quaternion.setFromRotationMatrix(m4);
    viz.smpl.scale.setScalar(scale);
    viz.smpl.position.copy(t);
  }

  function fitAlignFromMappingFixedScale(fixedScale) {
    if (!current.opencap.markerMap || !current.smpl.positions) throw new Error("OpenCap/SMPL data not loaded yet");

    const src = [];
    const dst = [];
    for (const [markerName, m] of Object.entries(current.mapping)) {
      const marker = current.opencap.markerMap.get(markerName);
      if (!marker) continue;
      if (!m?.face || !m?.barycentric) continue;
      src.push(smplBaryPointLocal(m.face, m.barycentric));
      dst.push(marker.position.clone());
    }
    if (src.length < 3) throw new Error("Alignment requires at least 3 valid mappings (face + barycentric)");

    const { scale, R, t } = fitRigidTransformFixedScale(src, dst, fixedScale);
    const m4 = new THREE.Matrix4();
    m4.set(R[0][0], R[0][1], R[0][2], 0, R[1][0], R[1][1], R[1][2], 0, R[2][0], R[2][1], R[2][2], 0, 0, 0, 0, 1);
    viz.smpl.quaternion.setFromRotationMatrix(m4);
    viz.smpl.scale.setScalar(scale);
    viz.smpl.position.copy(t);
  }

  function autoFitC7KneesIfAvailable() {
    if (!current.smpl.data?.meta?.autofit_anchors) return false;
    if (!current.opencap.markerMap) return false;
    if (!current.smpl.positions) return false;

    const anchors = current.smpl.data.meta.autofit_anchors;
    const required = ["C7_study", "L_knee_study", "r_knee_study"];
    for (const k of required) {
      const v = anchors?.[k]?.vertex;
      if (typeof v !== "number") return false;
      if (!current.opencap.markerMap.get(k)) return false;
    }

    for (const k of required) {
      const v = anchors[k].vertex;
      const m = mappingFromVertex(v);
      if (!m) return false;
      current.mapping[k] = m;
    }

    rebuildMappedPoints();
    renderMarkerList();

    try {
      const locked = getLockedScaleOrNull();
      if (locked) fitAlignFromMappingFixedScale(locked);
      else fitAlignFromMapping();
      return true;
    } catch {
      return false;
    }
  }

  function fitCamera() {
    const points = [];
    if (current.opencap.markerMap) for (const e of current.opencap.markerMap.values()) points.push(e.mesh.getWorldPosition(new THREE.Vector3()));
    if (current.smpl.positions && current.smpl.mesh && current.smpl.mesh.visible) {
      const b = current.smpl.mesh.geometry.boundingBox || new THREE.Box3().setFromObject(current.smpl.mesh);
      points.push(b.min.clone());
      points.push(b.max.clone());
    }
    if (!points.length) return;
    fitCameraToBox(viz.camera, viz.controls, computeBounds(points));
  }

  function buildExportObject() {
    const smplRot = new THREE.Euler().setFromQuaternion(viz.smpl.quaternion, "XYZ");
    return {
      meta: {
        format: "opencap43_to_smpl_manual_v1",
        created_at: new Date().toISOString(),
        opencap_scene: elements.opencapScenePath.value.trim(),
        smpl_mesh: elements.smplMeshPath.value.trim(),
      },
      smpl_transform: {
        position: [viz.smpl.position.x, viz.smpl.position.y, viz.smpl.position.z],
        rotation_xyz_rad: [smplRot.x, smplRot.y, smplRot.z],
        scale: [viz.smpl.scale.x, viz.smpl.scale.y, viz.smpl.scale.z],
      },
      mapping: current.mapping,
    };
  }

  function exportJson() {
    const obj = buildExportObject();
    downloadText("opencap43_to_smpl_manual.json", JSON.stringify(obj, null, 2), "application/json");
  }

  function exportCsv() {
    const names = markerNames();
    const lines = ["marker_name,vertex,face_a,face_b,face_c,bary_u,bary_v,bary_w"];
    for (const name of names) {
      const m = current.mapping[name];
      if (!m) {
        lines.push(`${name},,,,,,,`);
        continue;
      }
      const face = m.face || ["", "", ""];
      const b = m.barycentric || ["", "", ""];
      lines.push(`${name},${m.vertex},${face[0]},${face[1]},${face[2]},${b[0]},${b[1]},${b[2]}`);
    }
    downloadText("opencap43_to_smpl_manual.csv", lines.join("\n"), "text/csv");
  }

  function importJsonFile(file) {
    const reader = new FileReader();
    reader.onload = () => {
      try {
        const obj = JSON.parse(String(reader.result || ""));
        current.mapping = obj?.mapping || {};
        const t = obj?.smpl_transform;
        if (t?.position?.length === 3) viz.smpl.position.set(t.position[0], t.position[1], t.position[2]);
        if (t?.rotation_xyz_rad?.length === 3) viz.smpl.rotation.set(t.rotation_xyz_rad[0], t.rotation_xyz_rad[1], t.rotation_xyz_rad[2], "XYZ");
        if (t?.scale?.length === 3) viz.smpl.scale.set(t.scale[0], t.scale[1], t.scale[2]);
        rebuildMappedPoints();
        renderMarkerList();
        setMeta("Imported mapping JSON");
      } catch (e) {
        setMeta(`Import failed: ${e}`);
      }
    };
    reader.readAsText(file);
  }

  async function loadAll() {
    const ocUrl = elements.opencapScenePath.value.trim();
    const smplUrl = elements.smplMeshPath.value.trim();
    if (!ocUrl || !smplUrl) return;

    setMeta("Loading...");
    clearAllScene();

    const [ocData, smplData] = await Promise.all([
      loadJson(ocUrl, "OpenCap scene JSON"),
      loadJson(smplUrl, "SMPL mesh JSON").catch((e) => {
        const hint =
          String(e).includes("404") || String(e).includes("NOT_FOUND")
            ? `${e}\n\nTip: you need to generate smpl_template_mesh.json first (see 0108addbio/smpl_mapper/README.md).`
            : String(e);
        throw new Error(hint);
      }),
    ]);

    current.opencap.data = ocData;
    current.smpl.data = smplData;

    // If smpl mesh provides a recommended scale-to-BSM, default to enabling it.
    const scaleToBsm = Number(smplData?.meta?.scale_to_bsm_c7_ground);
    if (Number.isFinite(scaleToBsm) && scaleToBsm > 0) {
      elements.lockScaleToBsm.disabled = false;
      if (elements.lockScaleToBsm.checked === false) elements.lockScaleToBsm.checked = true;
    } else {
      elements.lockScaleToBsm.checked = false;
      elements.lockScaleToBsm.disabled = true;
    }

    // OpenCap bodies + skeleton.
    const bodies = new Map();
    for (const [name, entry] of Object.entries(ocData.bodies || {})) {
      if (!entry?.translation) continue;
      bodies.set(name, new THREE.Vector3(entry.translation[0], entry.translation[1], entry.translation[2]));
    }
    current.opencap.skeleton = buildOpenCapSkeleton(viz.openCap, bodies, ocData.edges || []);
    current.opencap.skeleton.group.visible = Boolean(elements.showOpenCapSkeleton.checked);

    // OpenCap markers.
    const markerRadius = Number(elements.markerSize.value || "0.02");
    current.opencap.markers = buildOpenCapMarkers(viz.openCap, ocData.markers || [], markerRadius, elements.showOpenCapLabels.checked);
    current.opencap.markerMap = current.opencap.markers.markerMap;
    current.opencap.markers.group.visible = Boolean(elements.showOpenCapMarkers.checked);
    updateLabelVisibility();

    // Optional OpenCap mesh.
    const missing = [];
    current.opencap.mesh = await buildOpenCapMesh(viz.openCap, ocData, missing);
    current.opencap.mesh.group.visible = Boolean(elements.showOpenCapMesh.checked);

    // SMPL mesh.
    resetSmplTransform();
    const smpl = buildSmplMesh(viz.smpl, smplData);
  current.smpl.mesh = smpl.mesh;
  current.smpl.geom = smpl.geometry;
  current.smpl.material = smpl.material;
  current.smpl.positions = smpl.positions;
  current.smpl.index = smpl.index;
  current.smpl.mesh.visible = Boolean(elements.showSmplMesh.checked);
  current.smpl.material.wireframe = Boolean(elements.showSmplWireframe.checked);

    // SMPL markers (pre-aligned) if available
    await loadSmplMarkers(viz.smpl, markerRadius);

  // SMPL pick marker + mapped dots.
  const pickSphere = new THREE.Mesh(new THREE.SphereGeometry(1, 12, 10), new THREE.MeshStandardMaterial({ color: 0xffbf66 }));
  pickSphere.scale.setScalar(0.016);
  pickSphere.visible = false;
    pickSphere.name = "smplPick";
    viz.smpl.add(pickSphere);
    current.smpl.pick = pickSphere;

    const mappedGroup = new THREE.Group();
    mappedGroup.name = "smplMapped";
    viz.smpl.add(mappedGroup);
    current.smpl.mappedGroup = mappedGroup;
    rebuildMappedPoints();

    viz.grid.visible = Boolean(elements.showGrid.checked);
    viz.axes.visible = Boolean(elements.showAxes.checked);
    applyLayout(viz);

    // Meta.
    let meshCount = 0;
    for (const body of Object.values(ocData.bodies || {})) meshCount += (body?.attachedGeometries || []).length;
    const baseMeta = `OpenCap markers: ${(ocData.markers || []).length}  bodies: ${bodies.size}  edges: ${(ocData.edges || []).length}  VTP meshes: ${meshCount}\nSMPL vertices: ${
      (smplData.vertices || []).length / 3
    }  faces: ${(smplData.faces || smplData.triangles || []).length / 3}`;
    const miss = missing.length ? `\nOpenCap mesh: missing ${missing.length} VTP files (e.g., ${missing.slice(0, 4).join(", ")})` : "";
    const didAutoFit = Object.keys(current.mapping || {}).length === 0 && autoFitC7KneesIfAvailable();
    const autoMsg = didAutoFit
      ? "\nAuto-fit applied: scaled by BSM C7-to-ground (if available), then fitted with C7_study + L_knee_study + r_knee_study"
      : "";
    setMeta(baseMeta + miss + autoMsg);

    renderMarkerList();
    fitCamera();
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
      if (dx * dx + dy * dy > 25) return;

      const picked = pickFromPointer(event);
      if (picked.marker) {
        setSelectedMarker(picked.marker);
        return;
      }
      if (picked.smpl) {
        const pick = buildPickFromSmplHit(picked.smpl);
        if (!pick) return;
        setSmplPick(pick);
        assignMappingForSelectedMarker(pick);
      }
    },
    { capture: true },
  );

  elements.load.addEventListener("click", () => loadAll().catch((e) => setMeta(String(e))));
  elements.exportJson.addEventListener("click", exportJson);
  elements.exportCsv.addEventListener("click", exportCsv);
  elements.importJson.addEventListener("change", () => {
    const f = elements.importJson.files?.[0];
    if (f) importJsonFile(f);
    elements.importJson.value = "";
  });

  elements.markerFilter.addEventListener("input", applyMarkerFilter);
  elements.clearCurrent.addEventListener("click", clearCurrent);
  elements.clearAll.addEventListener("click", clearAll);

  elements.layoutMode.addEventListener("change", () => {
    applyLayout(viz);
    fitCamera();
  });
  elements.splitDistance.addEventListener("input", () => {
    applyLayout(viz);
  });

  elements.smplYawPreset.addEventListener("change", () => {
    setSmplYawPreset();
  });
  elements.resetSmpl.addEventListener("click", () => {
    resetSmplTransform();
    rebuildMappedPoints();
  });
  elements.bboxAlign.addEventListener("click", () => {
    try {
      setSmplYawPreset();
      roughAlignBBox();
      fitCamera();
    } catch (e) {
      setMeta(String(e));
    }
  });
  elements.fitAlign.addEventListener("click", () => {
    try {
      const locked = getLockedScaleOrNull();
      if (locked) fitAlignFromMappingFixedScale(locked);
      else fitAlignFromMapping();
      fitCamera();
    } catch (e) {
      setMeta(String(e));
    }
  });
  elements.fitCamera.addEventListener("click", fitCamera);

  elements.showOpenCapMesh.addEventListener("change", () => {
    if (current.opencap.mesh?.group) current.opencap.mesh.group.visible = Boolean(elements.showOpenCapMesh.checked);
  });
  elements.showOpenCapSkeleton.addEventListener("change", () => {
    if (current.opencap.skeleton?.group) current.opencap.skeleton.group.visible = Boolean(elements.showOpenCapSkeleton.checked);
  });
  elements.showOpenCapMarkers.addEventListener("change", () => {
    if (current.opencap.markers?.group) current.opencap.markers.group.visible = Boolean(elements.showOpenCapMarkers.checked);
  });
  elements.showOpenCapLabels.addEventListener("change", updateLabelVisibility);

  elements.showSmplMesh.addEventListener("change", () => {
    if (current.smpl.mesh) current.smpl.mesh.visible = Boolean(elements.showSmplMesh.checked);
  });
  elements.showSmplWireframe.addEventListener("change", () => {
    if (current.smpl.material) current.smpl.material.wireframe = Boolean(elements.showSmplWireframe.checked);
  });
  elements.showMappedPoints.addEventListener("change", () => {
    if (current.smpl.mappedGroup) current.smpl.mappedGroup.visible = Boolean(elements.showMappedPoints.checked);
  });

  elements.showGrid.addEventListener("change", () => {
    viz.grid.visible = Boolean(elements.showGrid.checked);
  });
  elements.showAxes.addEventListener("change", () => {
    viz.axes.visible = Boolean(elements.showAxes.checked);
  });
  elements.markerSize.addEventListener("input", () => {
    if (!current.opencap.markerMap) return;
    const r = Number(elements.markerSize.value || "0.02");
    for (const entry of current.opencap.markerMap.values()) {
      entry.mesh.scale.setScalar(r);
      entry.label.position.set(0, r * 1.2, 0);
    }
  });

  elements.lockScaleToBsm?.addEventListener("change", () => {
    // Apply the locked scale immediately (does not auto-fit rotation/translation).
    const locked = getLockedScaleOrNull();
    if (locked) viz.smpl.scale.setScalar(locked);
  });

  setMeta("Loading...");
  applyLayout(viz);
  loadAll().catch((e) => setMeta(String(e)));
}

start();
