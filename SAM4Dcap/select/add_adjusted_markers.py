#!/usr/bin/env python3
"""
Generate a scene with adjusted thigh/shank cluster markers (right side searched, left side mirrored).
- Uses 7-anchor Procrustes (R+t+scale) to align SMPL to OpenCap (same as align_smpl_to_markers all mode).
- Searches nearest SMPL vertex (restricted to limb region) for right-side targets, mirrors to left via sym_idxs.npy.
- Appends green markers *_adj_smpl to the 43-marker compare scene.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import smplx
import torch
import yaml
from scipy.spatial import cKDTree

# Absolute paths for this packaged project
ROOT = Path("/root/TVB/SAM4Dcap/select")
DATA_DIR = ROOT / "data"
VIEWER_DIR = ROOT / "viewer"

# File defaults
DEF_SCENE = DATA_DIR / "scene_subject2_static1_frame1.json"
DEF_SMPL_NPZ = DATA_DIR / "smpl_results/00000000.npz"
DEF_SMPL_PKL = DATA_DIR / "SMPL_NEUTRAL_chumpy_free.pkl"
DEF_BSM_YAML = DATA_DIR / "bsm_markers.yaml"
DEF_SYM_NPY = DATA_DIR / "sym_idxs.npy"
DEF_BASE_COMPARE = DATA_DIR / "scene_subject2_static1_frame1_43markers_compare.json"

# Anchors for Procrustes (all mode)
BSM_VERTS = {"C7": 1306, "RKNE": 4495, "RKNI": 4634, "LKNE": 1010, "LKNI": 1148}
JOINT_INDEX = {"left_hip": 1, "right_hip": 2}
ANCHORS = [
    ("C7_study", "vertex", BSM_VERTS["C7"]),
    ("RHJC_study", "joint", JOINT_INDEX["right_hip"]),
    ("LHJC_study", "joint", JOINT_INDEX["left_hip"]),
    ("r_knee_study", "vertex", BSM_VERTS["RKNE"]),
    ("L_knee_study", "vertex", BSM_VERTS["LKNE"]),
    ("r_mknee_study", "vertex", BSM_VERTS["RKNI"]),
    ("L_mknee_study", "vertex", BSM_VERTS["LKNI"]),
]

# Right-side targets and their BSM centers for regional search
RIGHT_TARGETS = {
    "r_thigh1_study": "RFFT",
    "r_thigh2_study": "RFLT",
    "r_thigh3_study": "RFBT",
    "r_sh1_study": "RTIA",
    "r_sh2_study": "RTIB",
    "r_sh3_study": "RTIC",
}

# Right->Left marker name mapping
R2L = {
    "r_thigh1_study": "L_thigh1_study",
    "r_thigh2_study": "L_thigh2_study",
    "r_thigh3_study": "L_thigh3_study",
    "r_sh1_study": "L_sh1_study",
    "r_sh2_study": "L_sh2_study",
    "r_sh3_study": "L_sh3_study",
}


def load_scene(path: Path):
    data = json.loads(path.read_text())
    markers = {m["name"]: np.asarray(m["position"], float) for m in data.get("markers", [])}
    bodies = data.get("bodies", {})
    edges = data.get("edges", [])
    return markers, bodies, edges


def load_smpl(npz_path: Path, pkl_path: Path):
    data = np.load(npz_path, allow_pickle=True)
    verts = data["vertices"].astype(np.float64)
    betas = data["betas"].astype(np.float64)
    body_pose = data["body_pose"].astype(np.float64)
    global_orient = data["global_orient"].astype(np.float64)
    transl = data["transl"].astype(np.float64)
    model = smplx.SMPL(model_path=str(pkl_path), batch_size=1)
    with torch.no_grad():
        out = model(
            betas=torch.from_numpy(betas[None]).float(),
            body_pose=torch.from_numpy(body_pose[None]).float(),
            global_orient=torch.from_numpy(global_orient[None]).float(),
            transl=torch.from_numpy(transl[None]).float(),
        )
    joints = out.joints[0].cpu().numpy().astype(np.float64)
    faces = model.faces.reshape(-1).astype(int)
    return verts, joints, faces


def procrustes_similarity(src: np.ndarray, dst: np.ndarray):
    assert src.shape == dst.shape and src.shape[1] == 3
    n = src.shape[0]
    src_mean = src.mean(axis=0)
    dst_mean = dst.mean(axis=0)
    X = src - src_mean
    Y = dst - dst_mean
    cov = X.T @ Y / n
    U, S, Vt = np.linalg.svd(cov)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    scale = (n * S.sum()) / np.square(X).sum()
    t = dst_mean - scale * (R @ src_mean)
    return scale, R, t


def compute_transform(markers, verts, joints):
    src_pts = []
    dst_pts = []
    for name, kind, idx in ANCHORS:
        dst = markers[name]
        src = verts[idx] if kind == "vertex" else joints[idx]
        src_pts.append(src)
        dst_pts.append(dst)
    src_arr = np.stack(src_pts)
    dst_arr = np.stack(dst_pts)
    scale, R, t = procrustes_similarity(src_arr, dst_arr)

    def transform(points: np.ndarray) -> np.ndarray:
        return scale * (points @ R.T) + t

    return transform


def hip_frame(markers: Dict[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    rh = markers["RHJC_study"]
    lh = markers["LHJC_study"]
    c7 = markers["C7_study"]
    mid = 0.5 * (rh + lh)
    x = lh - rh
    x = x / (np.linalg.norm(x) + 1e-9)
    up = c7 - mid
    up = up / (np.linalg.norm(up) + 1e-9)
    z = np.cross(x, up)
    z = z / (np.linalg.norm(z) + 1e-9)
    y = np.cross(z, x)  # re-orthogonalize
    y = y / (np.linalg.norm(y) + 1e-9)
    # basis columns
    B = np.stack([x, y, z], axis=1)
    return mid, B


def nearest_in_region(
    tree: cKDTree, verts: np.ndarray, target: np.ndarray, center: np.ndarray, radii=(0.05, 0.1)
) -> Tuple[int, float]:
    for r in radii:
        idxs = tree.query_ball_point(center, r)
        if idxs:
            pts = verts[idxs]
            dists = np.linalg.norm(pts - target, axis=1)
            j = int(np.argmin(dists))
            return int(idxs[j]), float(dists[j])
    # fallback global nearest
    dist, idx = tree.query(target)
    return int(idx), float(dist)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", type=Path, default=DEF_SCENE)
    parser.add_argument("--smpl-npz", type=Path, default=DEF_SMPL_NPZ)
    parser.add_argument("--smpl-pkl", type=Path, default=DEF_SMPL_PKL)
    parser.add_argument("--bsm-yaml", type=Path, default=DEF_BSM_YAML)
    parser.add_argument("--sym-npy", type=Path, default=DEF_SYM_NPY)
    parser.add_argument(
        "--base-compare",
        type=Path,
        default=DEF_BASE_COMPARE,
        help="Existing 43-marker compare scene to extend with adjusted markers.",
    )
    parser.add_argument("--out-scene", type=Path, default=VIEWER_DIR / "scene_subject2_static1_frame1_43markers_compare_with_adj.json")
    args = parser.parse_args()

    # Load data
    markers_opc, bodies, edges = load_scene(args.scene)
    verts, joints, _ = load_smpl(args.smpl_npz, args.smpl_pkl)
    bsm_index = {k: int(v) for k, v in yaml.safe_load(Path(args.bsm_yaml).read_text()).items() if isinstance(v, (int, float))}
    sym = np.load(args.sym_npy)

    # Transform (all mode)
    transform = compute_transform(markers_opc, verts, joints)
    verts_aligned = transform(verts)
    joints_aligned = transform(joints)

    tree = cKDTree(verts_aligned)

    adjusted = []
    for r_name, bsm_center in RIGHT_TARGETS.items():
        if r_name not in markers_opc:
            print(f"[WARN] missing marker in scene: {r_name}")
            continue
        if bsm_center not in bsm_index:
            print(f"[WARN] BSM center not in YAML: {bsm_center}")
            continue
        p_hat = markers_opc[r_name]  # directly target the OpenCap marker position
        center_pos = verts_aligned[bsm_index[bsm_center]]
        vid, dist = nearest_in_region(tree, verts_aligned, p_hat, center_pos)
        adjusted.append((r_name, vid, dist, p_hat))

    # Build markers list from base compare scene (so original red/blue stay)
    base = json.loads(Path(args.base_compare).read_text())
    new_markers = base.get("markers", [])

    # Add adjusted right + mirrored left
    for r_name, vid, dist, p_hat in adjusted:
        pos_r = verts_aligned[vid]
        new_markers.append({"name": f"{r_name}_adj", "position": pos_r.tolist()})
        l_name = R2L.get(r_name)
        if l_name:
            vid_l = int(sym[vid])
            pos_l = verts_aligned[vid_l]
            new_markers.append({"name": f"{l_name}_adj", "position": pos_l.tolist()})

    out_scene = {
        "meta": {
            "note": "43 markers compare + adjusted thigh/shank clusters (green, _adj_smpl).",
            "source_scene": str(args.scene.resolve()),
            "source_smpl_npz": str(args.smpl_npz.resolve()),
            "sym_npy": str(args.sym_npy.resolve()),
        },
        "bodies": bodies,
        "edges": edges,
        "markers": new_markers,
    }

    args.out_scene.write_text(json.dumps(out_scene, indent=2))

    print(f"[OK] wrote {args.out_scene}")
    for r_name, vid, dist, _ in adjusted:
        print(f"{r_name:15s} -> vid {vid:4d}  dist={dist*100:.2f} cm  mirrored -> {R2L.get(r_name,'?')}")


if __name__ == "__main__":
    main()
