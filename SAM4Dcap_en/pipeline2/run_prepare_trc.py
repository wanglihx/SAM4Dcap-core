#!/usr/bin/env python3
"""
One-click generation of aligned 43-marker TRCs (dynamic/static) without touching code outside pipeline2.
Pipeline: sam-body4d -> MHRtoSMPL -> built-in alignment (camera extrinsics + height scaling + time/vertical alignment)
Outputs:
- Dynamic TRC: pipeline2/motion/subject2_43markers_aligned.trc
- Static TRC: pipeline2/static/static_43markers_aligned.trc
Intermediates are written to pipeline2/work_motion / work_static
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import yaml
from scipy.spatial.transform import Rotation as R
import smplx

# for TRC reader
sys.path.insert(0, "/root/TVB/opencap-core")
from utilsDataman import TRCFile  # noqa: E402

# Fixed Python interpreters to avoid wrong environments in subprocesses
BODY4D_PY = Path("/root/TVB/envs/body4d/bin/python")
MHR2SMPL_PY = Path("/root/TVB/envs/MHRtoSMPL/bin/python")


def run_cmd(cmd: list[str]) -> None:
    print("[CMD]", " ".join(cmd))
    subprocess.run(cmd, check=True)


def load_cam_params(cam_pickle: Path) -> Tuple[np.ndarray, np.ndarray]:
    import pickle

    with cam_pickle.open("rb") as f:
        cam = pickle.load(f)
    return cam["rotation"], cam["translation"].reshape(3, 1)  # R_cam, t_cam(mm)


def rotation_sequence(session_meta: Path) -> list[tuple[str, float]]:
    meta = yaml.safe_load(session_meta.read_text())
    placement = meta["checkerBoard"]["placement"]
    if placement in ("backWall", "Perpendicular"):
        return [("y", 90), ("z", 180)]
    elif placement in ("ground", "Lying"):
        return [("x", 90), ("y", 90)]
    raise ValueError(f"Unsupported checkerboard placement: {placement}")


def best_lag(a: List[float], b: List[float], max_lag: int = 90) -> int:
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    best = 0
    best_corr = -1.0
    for lag in range(-max_lag, max_lag + 1):
        if lag >= 0:
            a_seg = a[lag:]
            b_seg = b[: len(a_seg)]
        else:
            a_seg = a[: len(a) + lag]
            b_seg = b[-lag:]
        n = min(len(a_seg), len(b_seg))
        if n < 10:
            continue
        aa = a_seg[:n] - a_seg[:n].mean()
        bb = b_seg[:n] - b_seg[:n].mean()
        denom = np.linalg.norm(aa) * np.linalg.norm(bb) + 1e-8
        corr = float(np.dot(aa, bb) / denom)
        if corr > best_corr:
            best_corr = corr
            best = lag
    return best


def transform_cam_points(
    pts_cam_m: np.ndarray,
    scale: float,
    R_cam: np.ndarray,
    t_cam_mm: np.ndarray,
    rot_seq: list[tuple[str, float]],
) -> np.ndarray:
    cam_mm = pts_cam_m * scale * 1000.0
    world_mm = (R_cam.T @ (cam_mm.T - t_cam_mm)).T
    world_m = world_mm / 1000.0
    for axis, angle in rot_seq:
        world_m = R.from_euler(axis, angle, degrees=True).apply(world_m)
    return world_m


def load_opencap_trc(trc_path: Path) -> Tuple[np.ndarray, List[str], float]:
    trc = TRCFile(trc_path)
    n = trc.num_markers
    pts = np.zeros((trc.num_frames, n, 3), dtype=np.float32)
    for i, name in enumerate(trc.marker_names):
        pts[:, i, 0] = trc.data[name + "_tx"]
        pts[:, i, 1] = trc.data[name + "_ty"]
        pts[:, i, 2] = trc.data[name + "_tz"]
    return pts, trc.marker_names, float(trc.data_rate)


def load_markers_yaml(yaml_path: Path) -> Tuple[List[str], List[Tuple[str, int]]]:
    data = yaml.safe_load(yaml_path.read_text())
    names: List[str] = []
    items: List[Tuple[str, int]] = []
    for name, cfg in data.items():
        names.append(name)
        items.append((cfg.get("type", "vertex"), int(cfg["index"])))
    return names, items


def smpl_verts_and_hips(model: smplx.SMPL, npz_path: Path) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    d = np.load(npz_path)
    with torch.no_grad():
        out = model(
            betas=torch.from_numpy(d["betas"][None]).float(),
            body_pose=torch.from_numpy(d["body_pose"][None]).float(),
            global_orient=torch.from_numpy(d["global_orient"][None]).float(),
            transl=torch.from_numpy(d["transl"][None]).float(),
        )
    verts = out.vertices[0].cpu().numpy().astype(np.float32)
    joints = out.joints[0].cpu().numpy().astype(np.float32)
    hips = {"RHJC": joints[2], "LHJC": joints[1]}  # smplx joint indices
    return verts, hips


def procrustes_similarity(src: np.ndarray, dst: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray]:
    assert src.shape == dst.shape and src.shape[1] == 3
    n = src.shape[0]
    src_mean = src.mean(axis=0)
    dst_mean = dst.mean(axis=0)
    X = src - src_mean
    Y = dst - dst_mean
    cov = X.T @ Y / n
    U, S, Vt = np.linalg.svd(cov)
    Rm = Vt.T @ U.T
    if np.linalg.det(Rm) < 0:
        Vt[-1, :] *= -1
        Rm = Vt.T @ U.T
    scale = (n * S.sum()) / np.square(X).sum()
    t = dst_mean - scale * (Rm @ src_mean)
    return scale, Rm, t


def write_trc(path: Path, names: List[str], points: List[np.ndarray], data_rate: float) -> None:
    num_frames = len(points)
    num_markers = len(names)
    header1 = f"PathFileType\t4\t(X/Y/Z)\t{path}"
    header2 = (
        "DataRate\tCameraRate\tNumFrames\tNumMarkers\tUnits\tOrigDataRate\tOrigDataStartFrame\tOrigNumFrames"
    )
    header3 = f"{data_rate:.1f}\t{data_rate:.1f}\t{num_frames}\t{num_markers}\tm\t{data_rate:.1f}\t1\t{num_frames}"
    name_fields = []
    for n in names:
        name_fields.extend([n, "", ""])
    header4 = "\t".join(["Frame#", "Time"] + name_fields)
    coord_fields = []
    for i in range(num_markers):
        coord_fields.extend([f"X{i+1}", f"Y{i+1}", f"Z{i+1}"])
    header5 = "\t" + "\t".join(coord_fields)

    lines = [header1, header2, header3, header4, header5]
    for frame_idx, pts in enumerate(points, start=1):
        t = (frame_idx - 1) / data_rate
        vals = [f"{frame_idx}", f"{t:.7f}"]
        for p in pts:
            vals.extend([f"{p[0]:.7f}", f"{p[1]:.7f}", f"{p[2]:.7f}"])
        lines.append("\t".join(vals))
    path.write_text("\n".join(lines))


def infer_cam_pickle(video_path: Path, default: Path) -> Path:
    """Try to infer cam pickle from video path containing .../Videos/CamX/..."""
    parts = list(video_path.resolve().parents)
    for p in parts:
        if p.name.lower().startswith("cam") and len(p.name) > 3:
            cand = p / "cameraIntrinsicsExtrinsics.pickle"
            if cand.exists():
                return cand
    raise FileNotFoundError(
        f"Could not infer camera extrinsics from video path; please specify --cam-pickle(-dynamic/-static): {video_path}"
    )


def process_video(
    video_path: Path,
    workdir: Path,
    output_trc: Path,
    do_time_align: bool,
    opencap_trc: Path,
    cam_pickle: Path,
    session_meta: Path,
    markers_yaml: Path,
    smpl_model_path: Path,
) -> None:
    workdir.mkdir(parents=True, exist_ok=True)

    # step1: sam-body4d
    run_batch = Path("/root/TVB/sam-body4d/scripts/run_batch.py")
    run_cmd(
        [
            str(BODY4D_PY),
            str(run_batch),
            "--input",
            str(video_path),
            "--output",
            str(workdir),
        ]
    )

    # step2: MHR -> SMPL
    mhr2smpl = Path("/root/TVB/MHRtoSMPL/convert_mhr_to_smpl.py")
    mhr_params_dir = workdir / "mhr_params"
    smpl_results_dir = workdir / "smpl_results"
    run_cmd(
        [
            str(MHR2SMPL_PY),
            str(mhr2smpl),
            "--input",
            str(mhr_params_dir),
            "--output",
            str(smpl_results_dir),
        ]
    )

    # step3: align and write the 43-marker TRC
    rot_seq = rotation_sequence(session_meta)
    R_cam, t_cam = load_cam_params(cam_pickle)
    marker_names, marker_items = load_markers_yaml(markers_yaml)
    trc_opencap, _, frame_rate = load_opencap_trc(opencap_trc)

    mhr_files = sorted(mhr_params_dir.glob("*_data.npz"))
    smpl_files = sorted(smpl_results_dir.glob("*.npz"))
    if not mhr_files or not smpl_files:
        raise FileNotFoundError("Missing MHR or SMPL files")

    max_frames = min(len(mhr_files), len(smpl_files), trc_opencap.shape[0])

    smpl_model = smplx.SMPL(model_path=str(smpl_model_path), batch_size=1)

    first_smpl_body = np.load(smpl_files[0])["vertices"].astype(np.float32)
    meta = yaml.safe_load(session_meta.read_text())
    height_m = float(meta["height_m"])
    height_pred = first_smpl_body[:, 1].max() - first_smpl_body[:, 1].min()
    scale = float(height_m / height_pred)

    mhr_world_frames: List[np.ndarray] = []
    markers_world_frames: List[np.ndarray] = []
    mhr_min_y_raw: List[float] = []
    markers_min_y_raw: List[float] = []

    for i in range(max_frames):
        mhr_npz = np.load(mhr_files[i], allow_pickle=True)["data"].item()
        mhr_cam = mhr_npz["pred_vertices"].astype(np.float32) + mhr_npz["pred_cam_t"].astype(np.float32)
        mhr_world = transform_cam_points(mhr_cam, scale, R_cam, t_cam, rot_seq)
        mhr_world_frames.append(mhr_world)
        mhr_min_y_raw.append(float(mhr_world[:, 1].min()))

        smpl_npz = np.load(smpl_files[i])
        smpl_body = smpl_npz["vertices"].astype(np.float32)

        smpl_model_verts, hips_model = smpl_verts_and_hips(smpl_model, smpl_files[i])
        s_scale, s_R, s_t = procrustes_similarity(smpl_model_verts, smpl_body)
        hips_stored = {k: s_scale * (s_R @ v) + s_t for k, v in hips_model.items()}

        pts = []
        for typ, idx in marker_items:
            if typ == "vertex":
                pts.append(smpl_body[idx])
            elif typ == "joint":
                if idx == 2:
                    pts.append(hips_stored["RHJC"])
                elif idx == 1:
                    pts.append(hips_stored["LHJC"])
                else:
                    raise ValueError(f"Unsupported joint index: {idx}")
            else:
                raise ValueError(f"Unsupported marker type: {typ}")
        pts = np.stack(pts, axis=0) + mhr_npz["pred_cam_t"].astype(np.float32)
        pts_world = transform_cam_points(pts, scale, R_cam, t_cam, rot_seq)
        markers_world_frames.append(pts_world)
        markers_min_y_raw.append(float(pts_world[:, 1].min()))

    op_min_y_raw: List[float] = [float(trc_opencap[i][:, 1].min()) for i in range(trc_opencap.shape[0])]

    lag_mhr = best_lag(mhr_min_y_raw, op_min_y_raw, max_lag=90) if do_time_align else 0
    lag_all = lag_mhr

    sam_start = max(0, lag_all)
    op_start = max(0, -lag_all)
    n_frames = min(
        len(mhr_world_frames) - sam_start,
        len(markers_world_frames) - sam_start,
        trc_opencap.shape[0] - op_start,
    )

    mhr_world_frames = mhr_world_frames[sam_start : sam_start + n_frames]
    markers_world_frames = markers_world_frames[sam_start : sam_start + n_frames]

    mhr_min_y = []
    markers_min_y = []
    markers_points = []
    for i in range(n_frames):
        mhr_frame = mhr_world_frames[i]
        marker_frame = markers_world_frames[i]
        markers_points.append(marker_frame)
        mhr_min_y.append(float(mhr_frame[:, 1].min()))
        markers_min_y.append(float(marker_frame[:, 1].min()))

    offset_y = float(-min(np.min(mhr_min_y), np.min(markers_min_y)))
    for frame in markers_points:
        frame[:, 1] += offset_y

    write_trc(output_trc, marker_names, markers_points, frame_rate)
    print(
        f"[OK] wrote aligned TRC ({'time-aligned' if do_time_align else 'no time align'}) "
        f"frames={n_frames}, lag={lag_all}, offset_y={offset_y:.4f}, scale={scale:.4f}: {output_trc}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="From video to aligned 43-marker TRC (dynamic + static)")
    parser.add_argument("--dynamic-video", type=Path, required=True, help="Dynamic video path")
    parser.add_argument("--static-video", type=Path, help="Static video path (optional when --skip-static)")
    parser.add_argument(
        "--skip-static",
        action="store_true",
        help="Process dynamic video only; skip static workflow",
    )
    parser.add_argument(
        "--opencap-trc",
        type=Path,
        default=Path(
            "/root/TVB/SAM4Dcap/opencap/output/Data/subject2_Session0/MarkerData/mmpose_0.8/2-cameras/PostAugmentation_v0.2/DJ1_syncdWithMocap_LSTM.trc"
        ),
        help="OpenCap reference TRC (for time alignment)",
    )
    parser.add_argument(
        "--opencap-trc-dynamic",
        type=Path,
        help="OpenCap reference TRC used for the dynamic video (defaults to --opencap-trc)",
    )
    parser.add_argument(
        "--opencap-trc-static",
        type=Path,
        help="OpenCap reference TRC for the static video (defaults to --opencap-trc; static path only used for frame rate, no time alignment)",
    )
    parser.add_argument(
        "--cam-pickle",
        type=Path,
        default=Path("/root/TVB/SAM4Dcap/opencap/data/subject2/Session0/Videos/Cam1/cameraIntrinsicsExtrinsics.pickle"),
        help="Camera intrinsics/extrinsics pickle",
    )
    parser.add_argument(
        "--cam-pickle-dynamic",
        type=Path,
        help="Dynamic video camera extrinsics (if omitted, try inferring CamX from the video path; otherwise use --cam-pickle)",
    )
    parser.add_argument(
        "--cam-pickle-static",
        type=Path,
        help="Static video camera extrinsics (if omitted, try inferring CamX from the video path; otherwise use --cam-pickle)",
    )
    parser.add_argument(
        "--session-meta",
        type=Path,
        default=Path("/root/TVB/SAM4Dcap/opencap/data/subject2/Session0/sessionMetadata.yaml"),
        help="sessionMetadata.yaml containing height and checkerboard placement",
    )
    parser.add_argument(
        "--markers-yaml",
        type=Path,
        default=Path("/root/TVB/SAM4Dcap/align_43marks/markers_lai_43.yaml"),
        help="43-marker mapping YAML",
    )
    parser.add_argument(
        "--smpl-model",
        type=Path,
        default=Path("/root/TVB/MHRtoSMPL/SMPL_NEUTRAL_chumpy_free.pkl"),
        help="SMPL model path",
    )
    parser.add_argument(
        "--work-motion",
        type=Path,
        default=Path("/root/TVB/SAM4Dcap/pipeline2/work_motion"),
        help="Dynamic processing workspace directory",
    )
    parser.add_argument(
        "--work-static",
        type=Path,
        default=Path("/root/TVB/SAM4Dcap/pipeline2/work_static"),
        help="Static processing workspace directory",
    )
    parser.add_argument(
        "--out-motion",
        type=Path,
        default=Path("/root/TVB/SAM4Dcap/pipeline2/motion/subject2_43markers_aligned.trc"),
        help="Dynamic TRC output path",
    )
    parser.add_argument(
        "--out-static",
        type=Path,
        default=Path("/root/TVB/SAM4Dcap/pipeline2/static/static_43markers_aligned.trc"),
        help="Static TRC output path",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    cam_dyn = args.cam_pickle_dynamic or infer_cam_pickle(args.dynamic_video, args.cam_pickle)
    cam_sta = args.cam_pickle_static or (infer_cam_pickle(args.static_video, args.cam_pickle) if args.static_video else None)
    op_trc_dyn = args.opencap_trc_dynamic or args.opencap_trc
    op_trc_sta = args.opencap_trc_static or args.opencap_trc

    process_video(
        video_path=args.dynamic_video,
        workdir=args.work_motion,
        output_trc=args.out_motion,
        do_time_align=True,
        opencap_trc=op_trc_dyn,
        cam_pickle=cam_dyn,
        session_meta=args.session_meta,
        markers_yaml=args.markers_yaml,
        smpl_model_path=args.smpl_model,
    )

    if args.skip_static:
        print("[INFO] skip-static enabled; static video is not processed.")
        return

    if not args.static_video:
        raise ValueError("Missing --static-video (disable static processing with --skip-static)")

    process_video(
        video_path=args.static_video,
        workdir=args.work_static,
        output_trc=args.out_static,
        do_time_align=False,
        opencap_trc=op_trc_sta,
        cam_pickle=cam_sta,
        session_meta=args.session_meta,
        markers_yaml=args.markers_yaml,
        smpl_model_path=args.smpl_model,
    )


if __name__ == "__main__":
    main()
