import * as THREE from "https://unpkg.com/three@0.160.0/build/three.module.js";
import { OrbitControls } from "https://unpkg.com/three@0.160.0/examples/jsm/controls/OrbitControls.js";

// Server root is all_line/opencap, data lives under output/Data
const DATA_PREFIX = "/output/Data";

const DEFAULT_EDGES = [
  ["pelvis", "torso"],
  ["pelvis", "femur_r"],
  ["femur_r", "tibia_r"],
  ["tibia_r", "talus_r"],
  ["talus_r", "calcn_r"],
  ["calcn_r", "toes_r"],
  ["pelvis", "femur_l"],
  ["femur_l", "tibia_l"],
  ["tibia_l", "talus_l"],
  ["talus_l", "calcn_l"],
  ["calcn_l", "toes_l"],
  ["torso", "humerus_r"],
  ["humerus_r", "ulna_r"],
  ["humerus_r", "radius_r"],
  ["radius_r", "hand_r"],
  ["ulna_r", "hand_r"],
  ["torso", "humerus_l"],
  ["humerus_l", "ulna_l"],
  ["humerus_l", "radius_l"],
  ["radius_l", "hand_l"],
  ["ulna_l", "hand_l"],
];

const elements = {
  session: document.getElementById("session"),
  trial: document.getElementById("trial"),
  load: document.getElementById("load"),
  play: document.getElementById("play"),
  frame: document.getElementById("frame"),
  speed: document.getElementById("speed"),
  showMesh: document.getElementById("showMesh"),
  showSkeleton: document.getElementById("showSkeleton"),
  info: document.getElementById("info"),
  viewer: document.getElementById("viewer"),
  videos: document.getElementById("videos"),
};

const geometryCache = new Map();

function getUrlParams() {
  const params = new URLSearchParams(window.location.search);
  const session = params.get("session");
  const trial = params.get("trial");
  if (session) elements.session.value = session;
  if (trial) elements.trial.value = trial;
}

async function headOk(url) {
  try {
    const res = await fetch(url, { method: "HEAD" });
    return res.ok;
  } catch {
    return false;
  }
}

async function loadTrial(session, trial) {
  const jsonUrl = `${DATA_PREFIX}/${encodeURIComponent(session)}/VisualizerJsons/${encodeURIComponent(trial)}/${encodeURIComponent(trial)}.json`;
  const res = await fetch(jsonUrl);
  if (!res.ok) {
    throw new Error(`Failed to fetch JSON: ${res.status} ${res.statusText} (${jsonUrl})`);
  }
  const data = await res.json();
  return { data, jsonUrl };
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
  if (geometryCache.has(filename)) return geometryCache.get(filename);

  const promise = (async () => {
    const url = `./Geometry/${encodeURIComponent(filename)}`;
    const res = await fetch(url);
    if (!res.ok) {
      throw new Error(`Failed to fetch VTP: ${res.status} ${res.statusText} (${url})`);
    }
    const text = await res.text();
    return parseVtpAscii(text);
  })();

  geometryCache.set(filename, promise);
  return promise;
}

function median(values) {
  if (!values.length) return 0;
  const sorted = [...values].sort((a, b) => a - b);
  const mid = Math.floor(sorted.length / 2);
  if (sorted.length % 2) return sorted[mid];
  return (sorted[mid - 1] + sorted[mid]) / 2;
}

function makeRenderer(container) {
  const renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setPixelRatio(window.devicePixelRatio || 1);
  renderer.setSize(container.clientWidth, container.clientHeight);
  container.replaceChildren(renderer.domElement);
  return renderer;
}

function resize(renderer, camera, container) {
  const w = container.clientWidth;
  const h = container.clientHeight;
  if (w === 0 || h === 0) return;
  renderer.setSize(w, h);
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
}

function buildVideoGrid(session, trial) {
  elements.videos.replaceChildren();
  const cams = ["Cam0", "Cam1", "Cam2", "Cam3", "Cam4"];
  const promises = cams.map(async (cam) => {
    const url = `${DATA_PREFIX}/${encodeURIComponent(session)}/VisualizerVideos/${encodeURIComponent(trial)}/${encodeURIComponent(trial)}_syncd_${cam}.mp4`;
    const ok = await headOk(url);
    if (!ok) return null;
    return { cam, url };
  });
  return Promise.all(promises).then((items) => {
    const valid = items.filter(Boolean);
    if (!valid.length) {
      const div = document.createElement("div");
      div.className = "empty";
      div.textContent = "No synced videos found (may be cleaned or not generated).";
      elements.videos.appendChild(div);
      return;
    }
    for (const { cam, url } of valid) {
      const cell = document.createElement("div");
      cell.className = "video-cell";

      const title = document.createElement("div");
      title.className = "video-title";
      title.textContent = cam;

      const video = document.createElement("video");
      video.controls = true;
      video.playsInline = true;
      video.preload = "metadata";
      video.src = url;

      cell.appendChild(title);
      cell.appendChild(video);
      elements.videos.appendChild(cell);
    }
  });
}

function buildScene(container) {
  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0x0b0f14);

  const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 1000);
  camera.position.set(2.5, 1.6, 2.5);

  const modelGroup = new THREE.Group();
  modelGroup.name = "modelGroup";
  scene.add(modelGroup);

  const light1 = new THREE.DirectionalLight(0xffffff, 1.1);
  light1.position.set(3, 6, 2);
  scene.add(light1);
  scene.add(new THREE.AmbientLight(0xffffff, 0.4));

  const grid = new THREE.GridHelper(10, 20, 0x2b3340, 0x1b2230);
  grid.position.set(0, 0, 0);
  scene.add(grid);

  const axes = new THREE.AxesHelper(0.25);
  scene.add(axes);

  const renderer = makeRenderer(container);
  const controls = new OrbitControls(camera, renderer.domElement);
  controls.target.set(1.0, 1.0, -0.6);
  controls.update();

  window.addEventListener("resize", () => resize(renderer, camera, container));
  resize(renderer, camera, container);

  return { scene, camera, renderer, controls, modelGroup };
}

function buildSkeletonObjects(scene) {
  const group = new THREE.Group();
  group.name = "skeletonGroup";

  const lineMaterial = new THREE.LineBasicMaterial({ color: 0x66c2ff, linewidth: 1 });

  const geometry = new THREE.BufferGeometry();
  const positions = new Float32Array(DEFAULT_EDGES.length * 2 * 3);
  geometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));

  const lines = new THREE.LineSegments(geometry, lineMaterial);
  group.add(lines);

  const pointsGeometry = new THREE.BufferGeometry();
  const pointPositions = new Float32Array(0);
  pointsGeometry.setAttribute("position", new THREE.BufferAttribute(pointPositions, 3));

  const pointsMaterial = new THREE.PointsMaterial({ color: 0xffcc66, size: 0.02 });
  const points = new THREE.Points(pointsGeometry, pointsMaterial);
  group.add(points);

  scene.add(group);

  return { group, lines, points };
}

function updateFrame(state, frameIndex) {
  const { bodyTranslations, bodyRotations, bodyNames, edges, linePositions, pointPositions, pointsAttr, lineAttr, bodyGroups } = state;

  for (let i = 0; i < edges.length; i++) {
    const [a, b] = edges[i];
    const ta = bodyTranslations.get(a)[frameIndex];
    const tb = bodyTranslations.get(b)[frameIndex];
    const base = i * 2 * 3;
    linePositions[base + 0] = ta[0];
    linePositions[base + 1] = ta[1];
    linePositions[base + 2] = ta[2];
    linePositions[base + 3] = tb[0];
    linePositions[base + 4] = tb[1];
    linePositions[base + 5] = tb[2];
  }
  lineAttr.needsUpdate = true;

  for (let i = 0; i < bodyNames.length; i++) {
    const name = bodyNames[i];
    const t = bodyTranslations.get(name)[frameIndex];
    const base = i * 3;
    pointPositions[base + 0] = t[0];
    pointPositions[base + 1] = t[1];
    pointPositions[base + 2] = t[2];
  }
  pointsAttr.needsUpdate = true;

  if (bodyGroups) {
    for (const name of bodyNames) {
      const g = bodyGroups.get(name);
      if (!g) continue;
      const t = bodyTranslations.get(name)[frameIndex];
      const r = bodyRotations.get(name)[frameIndex];
      g.position.set(t[0], t[1], t[2]);
      g.rotation.set(r[0], r[1], r[2], "XYZ");
    }
  }
}

function setInfo(text) {
  elements.info.textContent = text;
}

function setPlayButton(playing) {
  elements.play.textContent = playing ? "Pause" : "Play";
}

async function start() {
  getUrlParams();

  const viz = buildScene(elements.viewer);
  const objs = buildSkeletonObjects(viz.scene);

  let playback = {
    loaded: false,
    playing: false,
    lastTs: null,
    accumulatorMs: 0,
    frameIndex: 0,
    frameDurationsMs: [],
    defaultFrameMs: 33.33,
    state: null,
  };

  function renderLoop(ts) {
    requestAnimationFrame(renderLoop);

    if (playback.loaded && playback.playing && playback.state) {
      if (playback.lastTs == null) playback.lastTs = ts;
      const speed = Number(elements.speed.value || "1.0");
      const dt = (ts - playback.lastTs) * speed;
      playback.lastTs = ts;
      playback.accumulatorMs += dt;

      const nFrames = playback.state.nFrames;
      while (playback.accumulatorMs > (playback.frameDurationsMs[playback.frameIndex] ?? playback.defaultFrameMs)) {
        playback.accumulatorMs -= (playback.frameDurationsMs[playback.frameIndex] ?? playback.defaultFrameMs);
        playback.frameIndex = (playback.frameIndex + 1) % nFrames;
        elements.frame.value = String(playback.frameIndex);
        updateFrame(playback.state, playback.frameIndex);
        setInfo(`frame ${playback.frameIndex + 1}/${nFrames}  t=${playback.state.time[playback.frameIndex].toFixed(3)}s`);
      }
    }

    viz.renderer.render(viz.scene, viz.camera);
  }
  requestAnimationFrame(renderLoop);

  async function loadAndShow() {
    const session = elements.session.value.trim();
    const trial = elements.trial.value.trim();
    if (!session || !trial) return;

    setInfo("Loading...");
    playback.playing = false;
    setPlayButton(false);
    playback.lastTs = null;
    playback.accumulatorMs = 0;

    const { data, jsonUrl } = await loadTrial(session, trial);

    const bodyNames = Object.keys(data.bodies).sort();
    const bodyTranslations = new Map();
    const bodyRotations = new Map();
    for (const name of bodyNames) {
      bodyTranslations.set(name, data.bodies[name].translation);
      bodyRotations.set(name, data.bodies[name].rotation);
    }

    const edges = DEFAULT_EDGES.filter(([a, b]) => bodyTranslations.has(a) && bodyTranslations.has(b));

    const nFrames = data.time.length;
    elements.frame.max = String(Math.max(0, nFrames - 1));
    elements.frame.value = "0";

    const dtMs = [];
    for (let i = 0; i < data.time.length - 1; i++) {
      dtMs.push((data.time[i + 1] - data.time[i]) * 1000);
    }
    const defaultFrameMs = median(dtMs) || 33.33;
    playback.frameDurationsMs = dtMs;
    playback.defaultFrameMs = defaultFrameMs;

    // (Re)build point attribute size based on body count.
    const pointPositions = new Float32Array(bodyNames.length * 3);
    objs.points.geometry.setAttribute("position", new THREE.BufferAttribute(pointPositions, 3));
    const pointsAttr = objs.points.geometry.getAttribute("position");

    // Line positions depend on edge count.
    const linePositions = new Float32Array(edges.length * 2 * 3);
    objs.lines.geometry.setAttribute("position", new THREE.BufferAttribute(linePositions, 3));
    const lineAttr = objs.lines.geometry.getAttribute("position");

    playback.state = {
      time: data.time,
      nFrames,
      bodyNames,
      bodyTranslations,
      bodyRotations,
      edges,
      linePositions,
      pointPositions,
      pointsAttr,
      lineAttr,
      bodyGroups: null,
    };

    viz.modelGroup.visible = Boolean(elements.showMesh?.checked);
    objs.group.visible = Boolean(elements.showSkeleton?.checked);

    if (viz.modelGroup.visible) {
      setInfo("Loading model mesh... (first time may be slower)");
      const bodyGroups = new Map();
      viz.modelGroup.clear();

      const geometryFiles = new Set();
      for (const name of bodyNames) {
        for (const f of data.bodies[name].attachedGeometries || []) geometryFiles.add(f);
      }

      await Promise.all([...geometryFiles].map((f) => loadVtpGeometry(f).catch(() => null)));

      for (let i = 0; i < bodyNames.length; i++) {
        const name = bodyNames[i];
        const body = data.bodies[name];

        const group = new THREE.Group();
        group.name = `body:${name}`;

        const s = body.scaleFactors || [1, 1, 1];
        group.scale.set(s[0], s[1], s[2]);

        const color = new THREE.Color();
        color.setHSL((i / Math.max(1, bodyNames.length)) * 0.9, 0.5, 0.55);
        const material = new THREE.MeshStandardMaterial({ color, roughness: 0.85, metalness: 0.05 });

        for (const geomFile of body.attachedGeometries || []) {
          const geom = await loadVtpGeometry(geomFile).catch(() => null);
          if (!geom) continue;
          const mesh = new THREE.Mesh(geom, material);
          mesh.frustumCulled = false;
          group.add(mesh);
        }

        viz.modelGroup.add(group);
        bodyGroups.set(name, group);
      }

      playback.state.bodyGroups = bodyGroups;
    } else {
      viz.modelGroup.clear();
      playback.state.bodyGroups = null;
    }

    playback.loaded = true;
    playback.frameIndex = 0;
    updateFrame(playback.state, 0);

    const pelvis0 = bodyTranslations.get("pelvis")?.[0];
    if (pelvis0) {
      viz.controls.target.set(pelvis0[0], pelvis0[1], pelvis0[2]);
      viz.controls.update();
    }

    setInfo(`Loaded: ${jsonUrl}  (frames=${nFrames}, ~${Math.round(1000 / defaultFrameMs)}fps)`);

    await buildVideoGrid(session, trial);
  }

  elements.load.addEventListener("click", () => loadAndShow().catch((e) => setInfo(String(e))));
  elements.play.addEventListener("click", () => {
    if (!playback.loaded) return;
    playback.playing = !playback.playing;
    playback.lastTs = null;
    setPlayButton(playback.playing);
  });
  elements.showMesh?.addEventListener("change", () => {
    viz.modelGroup.visible = Boolean(elements.showMesh?.checked);
  });
  elements.showSkeleton?.addEventListener("change", () => {
    objs.group.visible = Boolean(elements.showSkeleton?.checked);
  });
  elements.frame.addEventListener("input", () => {
    if (!playback.loaded || !playback.state) return;
    playback.playing = false;
    setPlayButton(false);
    playback.lastTs = null;
    playback.accumulatorMs = 0;
    playback.frameIndex = Number(elements.frame.value);
    updateFrame(playback.state, playback.frameIndex);
    setInfo(`frame ${playback.frameIndex + 1}/${playback.state.nFrames}  t=${playback.state.time[playback.frameIndex].toFixed(3)}s`);
  });

  await loadAndShow().catch((e) => setInfo(String(e)));
}

start().catch((e) => setInfo(String(e)));
