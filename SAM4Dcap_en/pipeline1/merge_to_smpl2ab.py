import argparse
import glob
import os
import sys
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(
        description="Merge per-frame SMPL npz files into a single npz."
    )
    default_frames = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "0109work_final_subject2")
    )
    default_out = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "subject2_60fps_male.npz")
    )
    parser.add_argument(
        "frames_dir",
        nargs="?",
        default=default_frames,
        help="Directory containing per-frame npz files (default: %(default)s)",
    )
    parser.add_argument(
        "-o",
        "--output",
        default=default_out,
        help="Output merged npz path (default: %(default)s)",
    )
    parser.add_argument(
        "--gender",
        default="male",
        choices=["male", "female", "neutral"],
        help="SMPL gender (default: %(default)s)",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=60.0,
        help="mocap_framerate (default: %(default)s)",
    )
    parser.add_argument(
        "--to-156",
        action="store_true",
        help="When set, pad hand/face zeros to 156 dims; otherwise keep 72 dims",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    frames_dir = os.path.abspath(args.frames_dir)
    out_path = os.path.abspath(args.output)
    gender = args.gender
    fps = args.fps
    to_156 = args.to_156

    frame_files = sorted(glob.glob(os.path.join(frames_dir, "*.npz")))
    if not frame_files:
        raise FileNotFoundError(f"No per-frame npz files found; check directory: {frames_dir}")

    trans_list = []
    poses_list = []
    betas_ref = None
    for idx, f in enumerate(frame_files):
        d = np.load(f)
        try:
            transl = d["transl"]
            global_orient = d["global_orient"]
            body_pose = d["body_pose"]
            betas = d["betas"]
        except KeyError as e:
            raise KeyError(f"{f} is missing field: {e}") from e

        if betas_ref is None:
            betas_ref = betas.astype(np.float32, copy=False)
        else:
            if betas.shape == betas_ref.shape and not np.allclose(betas_ref, betas):
                print(f"Warning: {f} betas differ slightly from the first frame; using first-frame betas.", file=sys.stderr)

        pose = np.concatenate([global_orient, body_pose]).astype(np.float32, copy=False)
        if to_156:
            pose = np.pad(pose, (0, 84))

        trans_list.append(transl.astype(np.float32, copy=False))
        poses_list.append(pose)

    trans = np.stack(trans_list)
    poses = np.stack(poses_list)

    np.savez(out_path,
             trans=trans,
             poses=poses,
             betas=betas_ref,
             gender=gender,
             mocap_framerate=fps)

    print(f"Wrote {out_path}")
    print(f"Frames: {len(frame_files)}, poses dimension: {poses.shape[1]}")


if __name__ == "__main__":
    main()
