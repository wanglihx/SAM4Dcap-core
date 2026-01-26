#!/usr/bin/env python3
"""
Align a single-frame SMPL mesh to OpenCap 43 markers with 7 anchors (one-shot Procrustes),
export the aligned mesh JSON for the viewer, and write an anchors compare scene.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import smplx

# BSM marker vertices on SMPL (0-based indices)
BSM_VERTS = {
    "C7": 1306,
    "RKNE": 4495,  # lateral knee (right)
    "RKNI": 4634,  # medial knee (right)
    "LKNE": 1010,  # lateral knee (left)
    "LKNI": 1148,  # medial knee (left)
}

JOINT_INDEX = {
    "left_hip": 1,
    "right_hip": 2,
}

ANCHORS = [
    ("C7_study", "vertex", BSM_VERTS["C7"]),
    ("RHJC_study", "joint", JOINT_INDEX["right_hip"]),
    ("LHJC_study", "joint", JOINT_INDEX["left_hip"]),
    ("r_knee_study", "vertex", BSM_VERTS["RKNE"]),
    ("L_knee_study", "vertex", BSM_VERTS["LKNE"]),
    ("r_mknee_study", "vertex", BSM_VERTS["RKNI"]),
    ("L_mknee_study", "vertex", BSM_VERTS["LKNI"]),
]

# Absolute paths for this packaged project
ROOT = Path("/root/TVB/SAM4Dcap/select")
DATA_DIR = ROOT / "data"
VIEWER_DIR = ROOT / "viewer"
SMPL_MAPPER_MESH = VIEWER_DIR / "smpl_mapper" / "smpl_frame0_aligned_mesh.json"


def load_markers(scene_path: Path) -> Dict[str, np.ndarray]:
    data = json.loads(scene_path.read_text())
    markers = {}
    for m in data.get("markers", []):
        name = m.get("name")
        pos = np.asarray(m.get("position", [0, 0, 0]), dtype=np.float64)
        markers[name] = pos
    return markers


def load_bodies_edges(scene_path: Path):
    data = json.loads(scene_path.read_text())
    return data.get("bodies", {}), data.get("edges", [])


def load_smpl_outputs(npz_path: Path):
    data = np.load(npz_path, allow_pickle=True)
    vertices = data["vertices"].astype(np.float64)
    betas = data["betas"].astype(np.float64)
    body_pose = data["body_pose"].astype(np.float64)
    global_orient = data["global_orient"].astype(np.float64)
    transl = data["transl"].astype(np.float64)
    return vertices, betas, body_pose, global_orient, transl


def compute_joints(model_path: Path, betas, body_pose, global_orient, transl) -> np.ndarray:
    smpl = smplx.SMPL(model_path=str(model_path), batch_size=1)
    with torch.no_grad():
        out = smpl(
            betas=torch.from_numpy(betas[None]).float(),
            body_pose=torch.from_numpy(body_pose[None]).float(),
            global_orient=torch.from_numpy(global_orient[None]).float(),
            transl=torch.from_numpy(transl[None]).float(),
        )
    joints = out.joints[0].cpu().numpy()
    return joints


def procrustes_similarity(src: np.ndarray, dst: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray]:
    """Return scale, rotation (3x3), translation for src->dst."""
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
    # scale = trace(Y^T R X) / ||X||^2 ; with cov divided by n, trace(Y^T R X) = n * sum(S)
    scale = (n * S.sum()) / np.square(X).sum()
    t = dst_mean - scale * (R @ src_mean)
    return scale, R, t


def _rodrigues(axis: np.ndarray, angle: float) -> np.ndarray:
    axis = axis / (np.linalg.norm(axis) + 1e-9)
    x, y, z = axis
    K = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smpl-npz", type=Path, default=DATA_DIR / "smpl_results/00000000.npz")
    parser.add_argument("--scene", type=Path, default=DATA_DIR / "scene_subject2_static1_frame1.json")
    parser.add_argument("--smpl-pkl", type=Path, default=DATA_DIR / "SMPL_NEUTRAL_chumpy_free.pkl")
    parser.add_argument("--out-mesh", type=Path, default=VIEWER_DIR / "smpl_frame0_aligned_mesh.json")
    parser.add_argument(
        "--out-anchors-scene",
        type=Path,
        default=VIEWER_DIR / "scene_subject2_static1_frame1_anchors_compare.json",
        help="Anchors compare scene (_opencap blue / _smpl red).",
    )
    args = parser.parse_args()

    marker_map = load_markers(args.scene)
    bodies, edges = load_bodies_edges(args.scene)
    vertices, betas, body_pose, global_orient, transl = load_smpl_outputs(args.smpl_npz)
    joints = compute_joints(args.smpl_pkl, betas, body_pose, global_orient, transl)

    src_pts: List[np.ndarray] = []
    dst_pts: List[np.ndarray] = []
    used_names: List[str] = []
    for name, kind, idx in ANCHORS:
        if name not in marker_map:
            raise RuntimeError(f"Marker {name} not found in scene {args.scene}")
        dst = marker_map[name]
        if kind == "vertex":
            src = vertices[idx]
        elif kind == "joint":
            src = joints[idx]
        else:
            raise ValueError(kind)
        src_pts.append(src)
        dst_pts.append(dst)
        used_names.append(name)

    src_arr = np.stack(src_pts, axis=0)
    dst_arr = np.stack(dst_pts, axis=0)

    # Single mode: one-shot Procrustes with 7 anchors (R + t + scale)
    scale, R, t = procrustes_similarity(src_arr, dst_arr)

    def apply_transform(points: np.ndarray) -> np.ndarray:
        return scale * (points @ R.T) + t

    aligned_vertices = apply_transform(vertices)
    aligned_anchors = apply_transform(src_arr)
    errors = np.linalg.norm(aligned_anchors - dst_arr, axis=1)

    print("Anchors (after alignment) RMS={:.4f} m  max={:.4f} m".format(np.sqrt((errors ** 2).mean()), errors.max()))
    for name, err in zip(used_names, errors):
        print(f"  {name:15s}  err={err:.4f} m")

    # Build mesh JSON for smpl_mapper
    meta = {
        "source_smpl_npz": str(args.smpl_npz.resolve()),
        "source_scene": str(args.scene.resolve()),
        "anchors": used_names,
        "rms_error_m": float(np.sqrt((errors ** 2).mean())),
        "max_error_m": float(errors.max()),
        "fit_mode": "all_procrustes",
    }
    mesh = {
        "meta": meta,
        "vertices": aligned_vertices.reshape(-1).tolist(),
        "faces": smplx.SMPL(model_path=str(args.smpl_pkl), batch_size=1).faces.reshape(-1).astype(int).tolist(),
    }
    mesh_json = json.dumps(mesh)
    args.out_mesh.parent.mkdir(parents=True, exist_ok=True)
    args.out_mesh.write_text(mesh_json)
    print(f"[OK] wrote aligned mesh: {args.out_mesh}")

    # keep smpl_mapper copy in sync if present
    for extra in [SMPL_MAPPER_MESH]:
        extra.parent.mkdir(parents=True, exist_ok=True)
        extra.write_text(mesh_json)
        print(f"[OK] copied aligned mesh to: {extra}")

    # Export a small scene with only the 7 anchors (opencap vs smpl) for color-coded viewing.
    markers_scene = []
    for name, dst, aligned in zip(used_names, dst_arr, aligned_anchors):
        markers_scene.append({"name": f"{name}_opencap", "position": dst.tolist()})
        markers_scene.append({"name": f"{name}_smpl", "position": aligned.tolist()})

    anchors_scene = {
        "meta": {
            "note": "Compare 7 anchors: _opencap (blue) vs _smpl (red)",
            "source_scene": str(args.scene.resolve()),
            "source_smpl_npz": str(args.smpl_npz.resolve()),
        },
        "bodies": bodies,
        "edges": edges,
        "markers": markers_scene,
    }
    args.out_anchors_scene.write_text(json.dumps(anchors_scene, indent=2))
    print(f"[OK] wrote anchors compare scene: {args.out_anchors_scene}")


if __name__ == "__main__":
    main()
