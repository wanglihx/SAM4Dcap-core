"""
Render SMPL vertices to video using pyrender.
"""
import argparse
import os
import pickle
import numpy as np
import trimesh
import pyrender
import cv2
from tqdm import tqdm
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Render SMPL vertices to an MP4.")
    parser.add_argument(
        "--smpl_output_dir",
        type=str,
        default="/root/TVB/MHRtoSMPL/output",
        help="Directory containing SMPL .npz files exported by convert_mhr_to_smpl.py.",
    )
    parser.add_argument(
        "--mhr_params_dir",
        type=str,
        default="/root/TVB/test/mhr_params",
        help="Directory containing *_data.npz from the original MHR output (used for camera intrinsics).",
    )
    parser.add_argument(
        "--smpl_model_path",
        type=str,
        default="/root/TVB/MHRtoSMPL/SMPL_NEUTRAL_chumpy_free.pkl",
        help="Path to SMPL model file (only faces are used).",
    )
    parser.add_argument(
        "--output_video",
        type=str,
        default="/root/TVB/MHRtoSMPL/smpl_render.mp4",
        help="Where to save the rendered video.",
    )
    parser.add_argument(
        "--frames_dir",
        type=str,
        default=None,
        help="If set, also保存每帧渲染图片到该目录（文件名按序号）。",
    )
    parser.add_argument("--img_w", type=int, default=1920, help="Render width.")
    parser.add_argument("--img_h", type=int, default=1080, help="Render height.")
    parser.add_argument("--fps", type=float, default=30.0, help="Video FPS.")
    return parser.parse_args()


def main():
    args = parse_args()

    smpl_output_dir = Path(args.smpl_output_dir)
    mhr_params_dir = Path(args.mhr_params_dir)
    output_video = args.output_video
    img_w, img_h = args.img_w, args.img_h
    frames_dir = Path(args.frames_dir) if args.frames_dir else None

    if not smpl_output_dir.exists():
        raise FileNotFoundError(f"SMPL输出目录不存在: {smpl_output_dir}")
    if not mhr_params_dir.exists():
        raise FileNotFoundError(f"MHR参数目录不存在: {mhr_params_dir}")

    # Load SMPL faces
    with open(args.smpl_model_path, 'rb') as f:
        smpl_data = pickle.load(f, encoding='latin1')
    faces = smpl_data['f']

    # Load first frame to get camera params from MHR data
    mhr_files = sorted(mhr_params_dir.glob("*_data.npz"))
    if not mhr_files:
        raise FileNotFoundError(f"在 {mhr_params_dir} 下没有找到 *_data.npz")
    first_mhr = mhr_files[0]
    mhr_data = np.load(first_mhr, allow_pickle=True)
    obj = mhr_data['data'][0]
    focal_length = float(obj['focal_length'])
    pred_cam_t = obj['pred_cam_t']  # Camera translation

    print(f"使用相机参数来自: {first_mhr}")
    print(f"Focal length: {focal_length}")
    print(f"Camera translation: {pred_cam_t}")

    # Get all SMPL output files
    smpl_files = sorted(smpl_output_dir.glob("*.npz"))
    if not smpl_files:
        raise FileNotFoundError(f"在 {smpl_output_dir} 下没有找到 *.npz（SMPL顶点文件）")
    print(f"Found {len(smpl_files)} SMPL frames")

    # Create pyrender scene
    scene = pyrender.Scene(bg_color=[1.0, 1.0, 1.0, 1.0])

    # Camera intrinsics
    fx = fy = focal_length
    cx, cy = img_w / 2, img_h / 2
    camera = pyrender.IntrinsicsCamera(fx=fx, fy=fy, cx=cx, cy=cy, znear=0.1, zfar=10.0)

    # Camera pose - identity rotation, translate by pred_cam_t
    cam_pose = np.eye(4)
    cam_pose[:3, 3] = pred_cam_t
    scene.add(camera, pose=cam_pose)

    # Add light
    light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=3.0)
    scene.add(light, pose=cam_pose)

    # Create offscreen renderer
    renderer = pyrender.OffscreenRenderer(viewport_width=img_w, viewport_height=img_h)

    # Video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_video, fourcc, args.fps, (img_w, img_h))
    if frames_dir:
        frames_dir.mkdir(parents=True, exist_ok=True)

    mesh_node = None

    for idx, smpl_file in enumerate(tqdm(smpl_files, desc="Rendering")):
        # Load SMPL vertices
        data = np.load(smpl_file)
        vertices = np.asarray(data['vertices'])

        # Create mesh
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
        mesh.visual.vertex_colors = [200, 200, 200, 255]  # Gray color

        # Update mesh in scene
        if mesh_node is not None:
            scene.remove_node(mesh_node)

        pyrender_mesh = pyrender.Mesh.from_trimesh(mesh, smooth=True)
        mesh_node = scene.add(pyrender_mesh)

        # Render
        color, depth = renderer.render(scene)

        # Convert RGB to BGR for OpenCV
        color_bgr = cv2.cvtColor(color, cv2.COLOR_RGB2BGR)
        video_writer.write(color_bgr)
        if frames_dir:
            frame_name = f"{idx:08d}.png"
            cv2.imwrite(str(frames_dir / frame_name), color_bgr)

    video_writer.release()
    renderer.delete()
    print(f"\nSaved video to {output_video}")
    if frames_dir:
        print(f"Saved frames to {frames_dir}")

if __name__ == "__main__":
    main()
