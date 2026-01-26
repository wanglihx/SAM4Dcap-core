#!/usr/bin/env python3
import argparse
import os
from pathlib import Path
import shutil

import yaml


def _copy_file(src: Path, dst: Path) -> None:
    if not src.exists():
        raise FileNotFoundError(f"Source missing: {src}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare OpenCap session data (local copy, no symlinks).")
    parser.add_argument(
        "--dataset-root",
        default="/root/TVB/LabValidation/LabValidation_withVideos",
        help="Root of LabValidation dataset (default: %(default)s)",
    )
    parser.add_argument("--data-root", type=Path, help="Where to place local data copy (default: ../data)")
    parser.add_argument("--web-root", type=Path, help="Where web assets live (default: ../web)")
    parser.add_argument("--subject", default="subject2", help="Subject id (default: %(default)s)")
    parser.add_argument("--session", default="Session0", help="Session name (default: %(default)s)")
    parser.add_argument(
        "--static-basename",
        default="static1",
        help="Baseline trial basename (without extension) for static scaling (default: %(default)s)",
    )
    parser.add_argument(
        "--dj1-basename",
        default="DJ1_syncdWithMocap",
        help="Dynamic trial basename (without extension) for DJ1 (default: %(default)s)",
    )
    parser.add_argument(
        "--video-ext",
        default="avi",
        help="Video file extension (default: %(default)s)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_dir = Path(__file__).resolve().parent
    project_root = base_dir.parent
    data_root = args.data_root or project_root / "data"
    web_root = args.web_root or project_root / "web"
    dataset_root = Path(args.dataset_root)

    subject = args.subject
    session = args.session

    session_dir = Path(data_root) / subject / session
    if session_dir.exists():
        shutil.rmtree(session_dir)
    session_dir.mkdir(parents=True, exist_ok=True)

    # sessionMetadata.yaml (copy + patch OpenSim model).
    src_meta = dataset_root / subject / "sessionMetadata.yaml"
    dst_meta = session_dir / "sessionMetadata.yaml"
    meta = yaml.safe_load(src_meta.read_text())
    meta["openSimModel"] = "LaiUhlrich2022"
    dst_meta.write_text(yaml.safe_dump(meta, sort_keys=False))

    cams = [f"Cam{i}" for i in range(5)]
    # Map trial folder -> source video basename (without extension).
    trial_video_names = {
        "static1": args.static_basename,
        "DJ1": args.dj1_basename,
        "extrinsics": "extrinsics",
    }

    for cam in cams:
        src_cam_dir = dataset_root / subject / "VideoData" / session / cam
        dst_cam_dir = session_dir / "Videos" / cam

        # Camera parameters (already calibrated in LabValidation).
        _copy_file(
            src_cam_dir / "cameraIntrinsicsExtrinsics.pickle",
            dst_cam_dir / "cameraIntrinsicsExtrinsics.pickle",
        )

        # Videos (copy, no symlinks).
        for trial, src_basename in trial_video_names.items():
            src_video = src_cam_dir / trial / f"{src_basename}.{args.video_ext}"
            if not src_video.exists():
                continue
            _copy_file(
                src_video,
                dst_cam_dir / "InputMedia" / trial / f"{src_basename}.{args.video_ext}",
            )

        # OpenCap expects the neutral pose trial to be under InputMedia/neutral.
        static_dir = dst_cam_dir / "InputMedia" / "static1"
        neutral_dir = dst_cam_dir / "InputMedia" / "neutral"
        if static_dir.exists():
            if neutral_dir.exists():
                shutil.rmtree(neutral_dir)
            shutil.copytree(static_dir, neutral_dir)

    # Stage OpenSim geometry (vtp) for the local web visualizer.
    geometry_src = None
    conda_prefix = os.environ.get("CONDA_PREFIX")
    if conda_prefix:
        candidate = Path(conda_prefix) / "share/doc/OpenSim/Code/Python/OpenSenseExample/Geometry"
        if candidate.exists():
            geometry_src = candidate
    if geometry_src is None:
        candidate = Path("/root/TVB/envs/opencap/share/doc/OpenSim/Code/Python/OpenSenseExample/Geometry")
        if candidate.exists():
            geometry_src = candidate

    if geometry_src is not None:
        copied = 0
        for web_subdir in ["webviz", "webviz_markerset"]:
            geometry_dst = Path(web_root) / web_subdir / "Geometry"
            geometry_dst.mkdir(parents=True, exist_ok=True)
            for src in geometry_src.glob("*.vtp"):
                dst = geometry_dst / src.name
                shutil.copy2(src, dst)
                copied += 1

            # Some OpenSim geometry packs use femur_l/tibia_l naming, while the
            # LaiUhlrich2022 model references l_femur/l_tibia. Create aliases so
            # the web viewer can resolve meshes directly from VisualizerJsons.
            alias_map = {
                "l_femur.vtp": "femur_l.vtp",
                "r_femur.vtp": "femur_r.vtp",
                "l_tibia.vtp": "tibia_l.vtp",
                "r_tibia.vtp": "tibia_r.vtp",
            }
            for dst_name, src_name in alias_map.items():
                dst = geometry_dst / dst_name
                src = geometry_dst / src_name
                if dst.exists() or not src.exists():
                    continue
                shutil.copy2(src, dst)
                copied += 1
        print(f"Staged OpenSim Geometry for webviz: {web_root} (copied {copied} files)")
    else:
        print("Warning: OpenSim Geometry directory not found; webviz mesh rendering will be unavailable.")

    print(f"Prepared local data at: {session_dir}")
    print("Next: run `python code/run_subject2_session0_dj1.py`")


if __name__ == "__main__":
    main()
