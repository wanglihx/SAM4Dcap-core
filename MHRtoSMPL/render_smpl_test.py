"""
Test render a single SMPL frame using the same rendering approach as sam-body4d.
Run with: /root/TVB/envs/body4d/bin/python render_smpl_test.py
"""
import os
os.environ["PYOPENGL_PLATFORM"] = "egl"

import numpy as np
import cv2
import pickle
import pyrender
import trimesh

def render_smpl(vertices, faces, cam_t, focal_length, img_h, img_w,
                mesh_base_color=(0.65, 0.74, 0.86)):
    """
    Render SMPL mesh using the same approach as sam-body4d's Renderer.

    Args:
        vertices: SMPL vertices (6890, 3)
        faces: SMPL faces (F, 3)
        cam_t: Camera translation (3,)
        focal_length: Focal length
        img_h, img_w: Image dimensions
        mesh_base_color: RGB color tuple (0-1 range)

    Returns:
        Rendered image (H, W, 3) in float [0, 1]
    """
    # Copy vertices and create mesh
    verts = vertices.copy()

    # Create vertex colors (RGBA) - RGB in 0-1 range, convert to 0-255
    vertex_colors = np.zeros((verts.shape[0], 4), dtype=np.uint8)
    r, g, b = mesh_base_color[0], mesh_base_color[1], mesh_base_color[2]
    rgba = np.array([r * 255, g * 255, b * 255, 255], dtype=np.uint8)
    vertex_colors[:] = rgba

    # Create trimesh
    mesh = trimesh.Trimesh(
        vertices=verts,
        faces=faces,
        vertex_colors=vertex_colors
    )

    # Apply 180 degree rotation around X-axis (same as sam-body4d)
    rot = trimesh.transformations.rotation_matrix(np.radians(180), [1, 0, 0])
    mesh.apply_transform(rot)

    # Convert to pyrender mesh
    mesh = pyrender.Mesh.from_trimesh(mesh, smooth=True)

    # Create scene
    scene = pyrender.Scene(
        bg_color=[1.0, 1.0, 1.0, 0.0],  # White background
        ambient_light=(0.3, 0.3, 0.3)
    )
    scene.add(mesh, "mesh")

    # Camera setup - negate X translation (same as sam-body4d)
    camera_translation = cam_t.copy()
    camera_translation[0] *= -1.0

    camera_pose = np.eye(4)
    camera_pose[:3, 3] = camera_translation

    camera = pyrender.IntrinsicsCamera(
        fx=focal_length,
        fy=focal_length,
        cx=img_w / 2.0,
        cy=img_h / 2.0,
        zfar=1e12,
    )
    scene.add(camera, pose=camera_pose)

    # Add lights (simplified version)
    light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=2.0)
    light_pose = np.eye(4)
    light_pose[:3, 3] = [0, 0, 5]
    scene.add(light, pose=light_pose)

    # Add point light for better illumination
    point_light = pyrender.PointLight(color=[1.0, 1.0, 1.0], intensity=3.0)
    scene.add(point_light, pose=camera_pose)

    # Create offscreen renderer
    renderer = pyrender.OffscreenRenderer(
        viewport_width=img_w,
        viewport_height=img_h,
    )

    # Render
    color, depth = renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
    color = color.astype(np.float32) / 255.0
    renderer.delete()

    return color[:, :, :3]


def main():
    # Load SMPL model faces
    smpl_model_path = "/root/TVB/MHRtoSMPL/SMPL_NEUTRAL_chumpy_free.pkl"
    with open(smpl_model_path, 'rb') as f:
        smpl_data = pickle.load(f, encoding='latin1')
    smpl_faces = smpl_data['f']
    print(f"SMPL faces shape: {smpl_faces.shape}")

    # Load MHR original data for camera params
    mhr_data = np.load('/root/TVB/test/mhr_params/00000000_data.npz', allow_pickle=True)
    obj = mhr_data['data'][0]
    focal_length = float(obj['focal_length'])
    pred_cam_t = obj['pred_cam_t']
    mhr_verts = obj['pred_vertices']

    print(f"Focal length: {focal_length}")
    print(f"pred_cam_t: {pred_cam_t}")
    print(f"MHR vertices shape: {mhr_verts.shape}")
    print(f"MHR vertices range: X=[{mhr_verts[:, 0].min():.3f}, {mhr_verts[:, 0].max():.3f}], Y=[{mhr_verts[:, 1].min():.3f}, {mhr_verts[:, 1].max():.3f}], Z=[{mhr_verts[:, 2].min():.3f}, {mhr_verts[:, 2].max():.3f}]")

    # Load SMPL converted vertices
    smpl_data = np.load('/root/TVB/MHRtoSMPL/output/00000000.npz')
    smpl_verts = smpl_data['vertices']
    smpl_transl = smpl_data['transl']
    print(f"\nSMPL vertices shape: {smpl_verts.shape}")
    print(f"SMPL transl: {smpl_transl}")
    print(f"SMPL vertices range: X=[{smpl_verts[:, 0].min():.3f}, {smpl_verts[:, 0].max():.3f}], Y=[{smpl_verts[:, 1].min():.3f}, {smpl_verts[:, 1].max():.3f}], Z=[{smpl_verts[:, 2].min():.3f}, {smpl_verts[:, 2].max():.3f}]")

    # Image size (same as MHR render)
    img_h, img_w = 1080, 1920

    # Render SMPL
    print("\n=== Rendering SMPL ===")
    smpl_render = render_smpl(
        smpl_verts, smpl_faces, pred_cam_t, focal_length, img_h, img_w,
        mesh_base_color=(0.65, 0.74, 0.86)  # light blue
    )

    # Save
    smpl_render_bgr = (smpl_render * 255).astype(np.uint8)[:, :, ::-1]  # RGB to BGR
    cv2.imwrite('/root/TVB/MHRtoSMPL/test_smpl_render.jpg', smpl_render_bgr)
    print("Saved: /root/TVB/MHRtoSMPL/test_smpl_render.jpg")

    # Check image stats
    non_white = np.sum(smpl_render_bgr < 250)
    print(f"Non-white pixels: {non_white}")

    print("\n=== Render Stats ===")
    print(f"SMPL render: min={smpl_render.min():.3f}, max={smpl_render.max():.3f}")

if __name__ == "__main__":
    main()
