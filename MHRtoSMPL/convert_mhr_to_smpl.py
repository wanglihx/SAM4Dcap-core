"""
MHR to SMPL Conversion Script

Converts MHR parameters from input directory to SMPL parameters in output directory.

Uses PyMomentum method with single_identity=True and is_tracking=True
"""

import os
import sys
import argparse  # 新增：用于解析命令行参数
from pathlib import Path

# Add MHR to path BEFORE any imports
sys.path.insert(0, '/root/TVB/MHR')
sys.path.insert(0, '/root/TVB/MHR/tools/mhr_smpl_conversion')

# IMPORTANT: Import pymomentum BEFORE torch to avoid segfault
import pymomentum.geometry
import pymomentum.torch

# Now import torch and other modules
import numpy as np
import torch
from tqdm import tqdm
import smplx
from mhr.mhr import MHR
from conversion import Conversion


def load_mhr_vertices(mhr_params_dir: str) -> np.ndarray:
    """Load all MHR vertices from the data files."""
    mhr_params_dir = Path(mhr_params_dir)

    # Find all data files
    data_files = sorted(mhr_params_dir.glob("*_data.npz"))
    print(f"Found {len(data_files)} frames in {mhr_params_dir}")

    if len(data_files) == 0:
        print(f"Error: No *_data.npz files found in {mhr_params_dir}")
        sys.exit(1)

    all_vertices = []
    for data_file in tqdm(data_files, desc="Loading MHR vertices"):
        data = np.load(data_file, allow_pickle=True)
        obj = data['data'][0]
        vertices = obj['pred_vertices']  # (18439, 3)
        all_vertices.append(vertices)

    # Stack to (N, 18439, 3)
    all_vertices = np.stack(all_vertices, axis=0)
    print(f"Loaded vertices shape: {all_vertices.shape}")

    return all_vertices


def main():
    # --- 新增：解析命令行参数 ---
    parser = argparse.ArgumentParser(description="Convert MHR to SMPL parameters")
    parser.add_argument('--input', type=str, default="/root/TVB/test/mhr_params", 
                        help='Input directory containing MHR params')
    parser.add_argument('--output', type=str, default="/root/TVB/MHRtoSMPL/output", 
                        help='Output directory for SMPL results')
    args = parser.parse_args()

    # 使用参数中的路径
    mhr_params_dir = args.input
    output_dir = args.output

    # 模型路径保持不变 (如果有需要也可以改成参数)
    smpl_model_path = "/root/TVB/MHRtoSMPL/SMPL_NEUTRAL_chumpy_free.pkl"
    mhr_assets_dir = "/root/TVB/MHR/assets"

    print(f"Input Directory:  {mhr_params_dir}")
    print(f"Output Directory: {output_dir}")
    
    os.makedirs(output_dir, exist_ok=True)

    # Device - PyMomentum only supports CPU
    device = torch.device("cpu")
    print(f"Using device: {device}")

    # Load MHR model
    print("Loading MHR model...")
    mhr_model = MHR.from_files(folder=Path(mhr_assets_dir), device=device, lod=1)

    # Load SMPL model
    print("Loading SMPL model...")
    smpl_model = smplx.SMPL(
        model_path=smpl_model_path,
    ).to(device)

    # Create converter with PyTorch method (PyMomentum only supports SMPL->MHR, not MHR->SMPL)
    print("Creating converter...")
    converter = Conversion(
        mhr_model=mhr_model,
        smpl_model=smpl_model,
        method="pytorch"  # MHR->SMPL only supports pytorch
    )

    # Load MHR vertices
    print("Loading MHR vertices...")
    mhr_vertices = load_mhr_vertices(mhr_params_dir)

    # Convert to tensor
    mhr_vertices_tensor = torch.from_numpy(mhr_vertices).float()

    # Run conversion
    print("Starting conversion...")
    print(f"  - Method: PyTorch (GPU if available)")
    print(f"  - Frames: {len(mhr_vertices)}")
    print(f"  - single_identity: True")

    results = converter.convert_mhr2smpl(
        mhr_vertices=mhr_vertices_tensor,
        mhr_parameters=None,
        single_identity=True,
        return_smpl_meshes=False,
        return_smpl_parameters=True,
        return_smpl_vertices=True,
        return_fitting_errors=True,
    )

    # Get results
    smpl_params = results.result_parameters
    smpl_vertices = results.result_vertices
    errors = results.result_errors

    print(f"\nConversion completed!")
    print(f"Fitting errors - mean: {errors.mean():.4f}, max: {errors.max():.4f}")

    # Save results
    print(f"\nSaving results to {output_dir}...")

    num_frames = len(mhr_vertices)

    for i in tqdm(range(num_frames), desc="Saving frames"):
        # Use MHR camera translation for global root motion when available.
        data_path = Path(mhr_params_dir) / f"{i:08d}_data.npz"
        mhr_data = np.load(data_path, allow_pickle=True)["data"].item()
        cam_t = mhr_data.get('pred_cam_t', None)

        frame_data = {
            'betas': smpl_params['betas'][i].detach().cpu().numpy() if 'betas' in smpl_params else smpl_params['betas'].detach().cpu().numpy(),
            'body_pose': smpl_params['body_pose'][i].detach().cpu().numpy(),
            'global_orient': smpl_params['global_orient'][i].detach().cpu().numpy(),
        }

        if cam_t is not None:
            # MHR uses camera coords (x right, y down, z forward); flip Y to make it y-up.
            cam_t_smpl = cam_t.astype(np.float32, copy=False)
            cam_t_smpl[1] *= 1.0
            frame_data['transl'] = cam_t_smpl
        elif 'transl' in smpl_params:
            frame_data['transl'] = smpl_params['transl'][i].detach().cpu().numpy()

        # Add vertices if available
        if smpl_vertices is not None:
            frame_data['vertices'] = smpl_vertices[i]

        # Add fitting error
        frame_data['fitting_error'] = errors[i]

        # Save
        output_path = os.path.join(output_dir, f"{i:08d}.npz")
        np.savez(output_path, **frame_data)

    print(f"\nDone! Saved {num_frames} frames to {output_dir}")


if __name__ == "__main__":
    main()
