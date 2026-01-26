#!/usr/bin/env python3
import argparse
import os
from pathlib import Path
import shutil
import sys


def _parse_cameras(value: str) -> list[str]:
    value = value.strip()
    if value.lower() == "all":
        return ["all"]
    cams = [v.strip() for v in value.split(",") if v.strip()]
    if not cams:
        raise ValueError("Camera list is empty.")
    return cams


def _stage_session(
    data_root: Path,
    output_root: Path,
    subject: str,
    session: str,
    session_name: str,
    reuse_output: bool,
) -> Path:
    src = data_root / subject / session
    if not src.exists():
        raise FileNotFoundError(f"Local data not found: {src} (run prepare script first)")

    dst = output_root / "Data" / session_name
    if dst.exists() and not reuse_output:
        shutil.rmtree(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(src, dst, dirs_exist_ok=True)
    return dst


def _run_phase(
    phase: str,
    *,
    session_name: str,
    data_dir: Path,
    camera_setup: str,
    pose_detector: str,
    bbox_thr: float,
    augmenter_model: str,
    static_trial: str,
    static_id: str,
    dj1_trial: str,
    dj1_id: str,
    cameras_static: list[str],
    cameras_dj1: list[str],
) -> None:
    opencap_core = "/root/TVB/opencap-core"
    sys.path.insert(0, opencap_core)
    from main import main as run_opencap  # noqa: E402

    if phase == "static":
        run_opencap(
            session_name,
            static_trial,
            static_id,
            cameras_to_use=cameras_static,
            poseDetector=pose_detector,
            bbox_thr=bbox_thr,
            generateVideo=False,
            scaleModel=True,
            augmenter_model=augmenter_model,
            markerDataFolderNameSuffix=camera_setup,
            dataDir=str(data_dir),
        )
        return

    if phase == "dj1":
        run_opencap(
            session_name,
            dj1_trial,
            dj1_id,
            cameras_to_use=cameras_dj1,
            poseDetector=pose_detector,
            bbox_thr=bbox_thr,
            generateVideo=False,
            scaleModel=False,
            augmenter_model=augmenter_model,
            markerDataFolderNameSuffix=camera_setup,
            dataDir=str(data_dir),
        )
        return

    raise ValueError(f"Unknown phase: {phase}")


def main() -> None:
    os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")

    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["all", "static", "dj1"], default="all", help="Run static and/or DJ1.")
    parser.add_argument("--subject", default="subject2")
    parser.add_argument("--session", default="Session0")
    parser.add_argument("--session-name", help="Override session folder name (default: <subject>_<session>).")
    parser.add_argument("--data-root", type=Path, help="Local data root (default: ../data)")
    parser.add_argument("--output-root", type=Path, help="Output root (default: ../output)")
    parser.add_argument("--camera-setup", default="2-cameras")
    parser.add_argument("--pose-detector", default="mmpose")
    parser.add_argument("--bbox-thr", type=float, default=0.8)
    parser.add_argument("--augmenter-model", default="v0.2")
    parser.add_argument("--static-trial", default="static1", help="Trial folder for static scaling.")
    parser.add_argument("--static-id", default="static1", help="Trial id (filename stem) for static scaling.")
    parser.add_argument("--dj1-trial", default="DJ1", help="Trial folder for DJ1.")
    parser.add_argument("--dj1-id", default="DJ1_syncdWithMocap", help="Trial id (filename stem) for DJ1.")
    parser.add_argument("--cameras-static", default="all", help="Comma-separated cameras for static (or 'all').")
    parser.add_argument("--cameras-dj1", default="Cam1,Cam3", help="Comma-separated cameras for DJ1.")
    parser.add_argument(
        "--reuse-output",
        action="store_true",
        help="Do not delete existing output/Data/<session> (stage data on top).",
    )
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parent
    project_root = base_dir.parent
    data_root = args.data_root or project_root / "data"
    output_root = args.output_root or project_root / "output"
    output_root.mkdir(parents=True, exist_ok=True)

    session_name = args.session_name or f"{args.subject}_{args.session}"
    cameras_static = _parse_cameras(args.cameras_static)
    cameras_dj1 = _parse_cameras(args.cameras_dj1)

    staged_dir = _stage_session(
        Path(data_root),
        Path(output_root),
        args.subject,
        args.session,
        session_name,
        args.reuse_output,
    )

    if args.phase in ("all", "static"):
        _run_phase(
            "static",
            session_name=session_name,
            data_dir=output_root,
            camera_setup=args.camera_setup,
            pose_detector=args.pose_detector,
            bbox_thr=args.bbox_thr,
            augmenter_model=args.augmenter_model,
            static_trial=args.static_trial,
            static_id=args.static_id,
            dj1_trial=args.dj1_trial,
            dj1_id=args.dj1_id,
            cameras_static=cameras_static,
            cameras_dj1=cameras_dj1,
        )

    if args.phase in ("all", "dj1"):
        _run_phase(
            "dj1",
            session_name=session_name,
            data_dir=output_root,
            camera_setup=args.camera_setup,
            pose_detector=args.pose_detector,
            bbox_thr=args.bbox_thr,
            augmenter_model=args.augmenter_model,
            static_trial=args.static_trial,
            static_id=args.static_id,
            dj1_trial=args.dj1_trial,
            dj1_id=args.dj1_id,
            cameras_static=cameras_static,
            cameras_dj1=cameras_dj1,
        )

    print(f"Done. Outputs are in: {staged_dir}")


if __name__ == "__main__":
    main()
