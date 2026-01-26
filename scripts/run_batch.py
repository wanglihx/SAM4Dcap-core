#!/usr/bin/env python
"""
SAM-Body4D 批量处理脚本
无需 Gradio 界面，命令行直接处理视频
自动检测人体并进行 4D 重建

使用方式:
    # 单个视频
    python scripts/run_batch.py --input /path/to/video.mp4 --output /path/to/output/

    # 批量处理目录
    python scripts/run_batch.py --input /path/to/videos/ --output /path/to/outputs/
"""

import os
import sys
import argparse
import glob
import time

# 使用 EGL 后端（GPU 加速渲染，renderer.py 默认使用 EGL）
# 注意：如果无 GPU 可用，需改回 'osmesa' 并升级 PyOpenGL

# 设置项目根目录
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.append(os.path.join(ROOT, 'models', 'sam_3d_body'))
sys.path.append(os.path.join(ROOT, 'models', 'diffusion_vas'))

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
from omegaconf import OmegaConf

from utils import (
    mask_painter, DAVIS_PALETTE, jpg_folder_to_mp4,
    keep_largest_component, is_super_long_or_wide, is_skinny_mask,
    bbox_from_mask, resize_mask_with_unique_label
)
from models.sam_3d_body.sam_3d_body import load_sam_3d_body, SAM3DBodyEstimator
from models.sam_3d_body.notebook.utils import process_image_with_mask, save_mesh_results
from models.sam_3d_body.tools.vis_utils import visualize_sample_together
from models.diffusion_vas.demo import init_amodal_segmentation_model, init_rgb_model, init_depth_model


# 设备选择
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"Using device: {device}")

if device.type == "cuda":
    # 启用 TF32 加速（与 app.py 保持一致）
    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True


class BatchProcessor:
    """批量处理器"""

    def __init__(self, config_path: str = None, detector_path: str = None):
        if config_path is None:
            config_path = os.path.join(ROOT, "configs", "body4d.yaml")

        print("=" * 60)
        print("SAM-Body4D Batch Processor")
        print("=" * 60)

        print("\n[1/4] Loading configuration...")
        self.config = OmegaConf.load(config_path)

        # 禁用遮挡恢复（加速处理，但仍加载 VAS 模型以测试）
        self.config.completion.enable = False
        print("  Occlusion recovery: DISABLED (for speed)")

        print("\n[2/4] Loading SAM-3 model...")
        self.sam3_model, self.predictor = self._build_sam3()

        print("\n[3/4] Loading SAM-3D-Body model...")
        self.sam3_3d_body_model = self._build_sam3_3d_body()

        print("\n[4/4] Loading Human Detector (ViTDet)...")
        self.human_detector = self._build_human_detector(detector_path)

        # 加载 VAS 模型（测试是否影响 SAM-3D-Body 行为）
        print("\n[Extra] Loading Diffusion-VAS models (for testing)...")
        self._load_vas_models()

        self.batch_size = self.config.sam_3d_body.get('batch_size', 64)
        print(f"\nBatch size: {self.batch_size}")
        print("=" * 60)
        print("All models loaded successfully!")
        print("=" * 60)

    def _build_sam3(self):
        """构建 SAM-3 模型"""
        from models.sam3.sam3.model_builder import build_sam3_video_model

        sam3_model = build_sam3_video_model(
            checkpoint_path=self.config.sam3['ckpt_path']
        )
        predictor = sam3_model.tracker
        predictor.backbone = sam3_model.detector.backbone

        return sam3_model, predictor

    def _build_sam3_3d_body(self):
        """构建 SAM-3D-Body 模型（与 app.py 保持一致）"""
        from models.sam_3d_body.tools.build_fov_estimator import FOVEstimator

        model, model_cfg = load_sam_3d_body(
            self.config.sam_3d_body['ckpt_path'],
            device=device,
            mhr_path=self.config.sam_3d_body['mhr_path']
        )

        fov_estimator = FOVEstimator(
            name='moge2',
            device=device,
            path=self.config.sam_3d_body['fov_path']
        )

        estimator = SAM3DBodyEstimator(
            sam_3d_body_model=model,
            model_cfg=model_cfg,
            human_detector=None,  # 我们单独使用检测器
            human_segmentor=None,
            fov_estimator=fov_estimator,
        )

        return estimator

    def _build_human_detector(self, detector_path: str = None):
        """构建人体检测器"""
        from models.sam_3d_body.tools.build_detector import HumanDetector

        if detector_path is None:
            detector_path = "/root/autodl-tmp/vitdet"

        detector = HumanDetector(
            name="vitdet",
            device=device,
            path=detector_path
        )
        return detector

    def _load_vas_models(self):
        """加载 Diffusion-VAS 模型（仅用于测试，不实际使用）"""
        model_path_mask = self.config.completion['model_path_mask']
        model_path_rgb = self.config.completion['model_path_rgb']
        model_path_depth = self.config.completion['model_path_depth']
        depth_encoder = self.config.completion['depth_encoder']

        # 加载模型但不保存引用（仅为了触发初始化）
        print("    Loading amodal segmentation model...")
        self.pipeline_mask = init_amodal_segmentation_model(model_path_mask)
        print("    Loading RGB completion model...")
        self.pipeline_rgb = init_rgb_model(model_path_rgb)
        print("    Loading depth model...")
        self.depth_model = init_depth_model(model_path_depth, depth_encoder)
        print("    VAS models loaded (not used, just for testing)")

    def _read_first_frame(self, video_path: str):
        """读取视频第一帧和帧率"""
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        ok, frame = cap.read()
        cap.release()
        if not ok:
            return None, fps
        # 返回 BGR 格式（检测器需要）和帧率
        return frame, fps

    def _detect_humans(self, frame):
        """检测人体，返回边界框列表"""
        boxes = self.human_detector.run_human_detection(frame)
        return boxes

    def process_video(self, video_path: str, output_dir: str):
        """处理单个视频"""
        print(f"\n{'='*60}")
        print(f"Processing: {video_path}")
        print(f"Output: {output_dir}")
        print(f"{'='*60}")

        os.makedirs(output_dir, exist_ok=True)

        # Step 1: 初始化视频状态
        print("\n[Step 1/6] Initializing video state...")
        inference_state = self.predictor.init_state(video_path=video_path)

        video_height = inference_state['video_height']
        video_width = inference_state['video_width']
        num_frames = len(inference_state['images'])
        print(f"  Video size: {video_width}x{video_height}, {num_frames} frames")

        # Step 2: 检测人体
        print("\n[Step 2/6] Detecting humans in first frame...")
        first_frame, video_fps = self._read_first_frame(video_path)
        if first_frame is None:
            print("  ERROR: Failed to read first frame")
            return None
        print(f"  Video FPS: {video_fps}")

        boxes = self._detect_humans(first_frame)
        print(f"  Detected {len(boxes)} person(s)")

        if len(boxes) == 0:
            print("  WARNING: No humans detected, using center point as fallback")
            # 回退到中心点
            boxes = np.array([[
                video_width * 0.25, video_height * 0.25,
                video_width * 0.75, video_height * 0.75
            ]])

        # Step 3: 为每个人添加框+点提示
        print("\n[Step 3/6] Adding box+point prompts for each person...")
        out_obj_ids = []
        for i, box in enumerate(boxes):
            # 计算边界框中心
            center_x = (box[0] + box[2]) / 2
            center_y = (box[1] + box[3]) / 2

            # 转换框为相对坐标
            rel_box = torch.tensor([
                box[0] / video_width,
                box[1] / video_height,
                box[2] / video_width,
                box[3] / video_height
            ], dtype=torch.float32)

            # 中心点相对坐标
            rel_x = center_x / video_width
            rel_y = center_y / video_height

            points_tensor = torch.tensor([[rel_x, rel_y]], dtype=torch.float32)
            points_labels_tensor = torch.tensor([1], dtype=torch.int32)

            obj_id = i + 1
            _, current_obj_ids, _, _ = self.predictor.add_new_points_or_box(
                inference_state=inference_state,
                frame_idx=0,
                obj_id=obj_id,
                box=rel_box,  # 添加框提示 (label 2, 3)
                points=points_tensor,  # 中心点 (label 1) 用于追踪
                labels=points_labels_tensor,
            )
            out_obj_ids = current_obj_ids
            print(f"  Person {i+1}: box=[{box[0]:.0f},{box[1]:.0f},{box[2]:.0f},{box[3]:.0f}], center=({center_x:.0f},{center_y:.0f})")

        print(f"  Object IDs: {out_obj_ids}")

        # Step 4: 视频传播分割
        print("\n[Step 4/6] Propagating segmentation...")
        video_segments = {}
        for frame_idx, obj_ids, low_res_masks, video_res_masks, obj_scores, iou_scores in tqdm(
            self.predictor.propagate_in_video(
                inference_state,
                start_frame_idx=0,
                max_frame_num_to_track=num_frames,
                reverse=False,
                propagate_preflight=True,
            ),
            total=num_frames,
            desc="  Propagating"
        ):
            video_segments[frame_idx] = {
                out_obj_id: (video_res_masks[i] > 0.0).cpu().numpy()
                for i, out_obj_id in enumerate(out_obj_ids)
            }

        # Step 5: 保存帧和掩码
        print("\n[Step 5/6] Saving frames and masks...")
        image_path = os.path.join(output_dir, 'images')
        masks_path = os.path.join(output_dir, 'masks')
        os.makedirs(image_path, exist_ok=True)
        os.makedirs(masks_path, exist_ok=True)

        for frame_idx in tqdm(range(len(video_segments)), desc="  Saving"):
            # 获取原始帧
            img = inference_state['images'][frame_idx].detach().float().cpu()
            img = (img + 1) / 2
            img = img.clamp(0, 1)
            img = F.interpolate(
                img.unsqueeze(0),
                size=(video_height, video_width),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)
            img = img.permute(1, 2, 0)
            img = (img.numpy() * 255).astype("uint8")
            img_pil = Image.fromarray(img).convert('RGB')

            # 构建掩码
            msk = np.zeros((video_height, video_width), dtype=np.uint8)
            for out_obj_id, out_mask in video_segments[frame_idx].items():
                mask = (out_mask[0] > 0).astype(np.uint8) * 255
                msk[mask == 255] = out_obj_id

            msk_pil = Image.fromarray(msk).convert('P')
            msk_pil.putpalette(DAVIS_PALETTE)

            img_pil.save(os.path.join(image_path, f"{frame_idx:08d}.jpg"))
            msk_pil.save(os.path.join(masks_path, f"{frame_idx:08d}.png"))

        # Step 6: 4D 重建
        print("\n[Step 6/6] Running 4D reconstruction...")
        self._run_4d_generation(output_dir, out_obj_ids)

        # 生成输出视频
        print("\n[Final] Generating output video...")
        out_video_path = os.path.join(output_dir, f"4d_{int(time.time())}.mp4")
        jpg_folder_to_mp4(os.path.join(output_dir, 'rendered_frames'), out_video_path, fps=int(video_fps))
        print(f"  Output video: {out_video_path}")

        print(f"\n{'='*60}")
        print(f"Completed: {video_path}")
        print(f"{'='*60}\n")

        return out_video_path

    def _run_4d_generation(self, output_dir: str, out_obj_ids: list):
        """运行 4D 生成"""
        image_path = os.path.join(output_dir, 'images')
        masks_path = os.path.join(output_dir, 'masks')

        image_extensions = ["*.jpg", "*.jpeg", "*.png"]
        images_list = sorted([
            img for ext in image_extensions
            for img in glob.glob(os.path.join(image_path, ext))
        ])
        masks_list = sorted([
            img for ext in image_extensions
            for img in glob.glob(os.path.join(masks_path, ext))
        ])

        os.makedirs(os.path.join(output_dir, 'rendered_frames'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'mhr_params'), exist_ok=True)
        for obj_id in out_obj_ids:
            os.makedirs(os.path.join(output_dir, f'mesh_4d_individual/{obj_id}'), exist_ok=True)

        n = len(images_list)
        batch_size = self.batch_size

        mhr_shape_scale_dict = {}

        for i in tqdm(range(0, n, batch_size), desc="  4D reconstruction"):
            batch_images = images_list[i:i + batch_size]
            batch_masks = masks_list[i:i + batch_size]

            # 创建空的 occ_dict（禁用遮挡恢复时）
            occ_dict = {obj_id: [1] * len(batch_masks) for obj_id in out_obj_ids}

            # 强制禁用 autocast，防止 MHR 内部的稀疏矩阵运算因 BFloat16 报错
            with torch.autocast("cuda", enabled=False):
                mask_outputs, id_batch, empty_frame_list = process_image_with_mask(
                    self.sam3_3d_body_model,
                    batch_images,
                    batch_masks,
                    {},  # idx_path (empty, no completion)
                    {},  # idx_dict (empty, no completion)
                    mhr_shape_scale_dict,
                    occ_dict
                )

            num_empty_ids = 0
            for frame_id in range(len(batch_images)):
                image_path_current = batch_images[frame_id]
                if frame_id in empty_frame_list:
                    mask_output = None
                    id_current = None
                    num_empty_ids += 1
                else:
                    mask_output = mask_outputs[frame_id - num_empty_ids]
                    id_current = id_batch[frame_id - num_empty_ids]

                img = cv2.imread(image_path_current)
                rend_img = visualize_sample_together(
                    img, mask_output, self.sam3_3d_body_model.faces, id_current
                )

                cv2.imwrite(
                    os.path.join(output_dir, 'rendered_frames',
                                f"{os.path.basename(image_path_current)[:-4]}.jpg"),
                    rend_img.astype(np.uint8),
                )

                # 保存参数
                np.savez_compressed(
                    os.path.join(output_dir, 'mhr_params',
                                f"{os.path.basename(image_path_current)[:-4]}_data.npz"),
                    data=mask_output
                )
                np.savez_compressed(
                    os.path.join(output_dir, 'mhr_params',
                                f"{os.path.basename(image_path_current)[:-4]}_id.npz"),
                    data=id_current
                )


def main():
    parser = argparse.ArgumentParser(
        description="SAM-Body4D 批量处理脚本（自动人体检测）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 单个视频
  python scripts/run_batch.py --input video.mp4 --output ./output/

  # 批量处理目录中的所有视频
  python scripts/run_batch.py --input ./videos/ --output ./outputs/
        """
    )
    parser.add_argument(
        '--input', '-i',
        required=True,
        help='输入视频路径或包含视频的目录'
    )
    parser.add_argument(
        '--output', '-o',
        required=True,
        help='输出目录'
    )
    parser.add_argument(
        '--config', '-c',
        default=None,
        help='配置文件路径 (默认: configs/body4d.yaml)'
    )
    parser.add_argument(
        '--detector-path',
        default='/root/autodl-tmp/vitdet',
        help='ViTDet 检测器模型路径 (默认: /root/autodl-tmp/vitdet)'
    )

    args = parser.parse_args()

    # 切换到项目目录
    os.chdir(ROOT)

    # 初始化处理器
    processor = BatchProcessor(
        config_path=args.config,
        detector_path=args.detector_path
    )

    # 判断输入是文件还是目录
    if os.path.isfile(args.input):
        # 单个视频
        processor.process_video(args.input, args.output)
    elif os.path.isdir(args.input):
        # 批量处理目录
        video_files = []
        for ext in ['*.mp4', '*.MP4', '*.avi', '*.AVI', '*.mov', '*.MOV']:
            video_files.extend(glob.glob(os.path.join(args.input, ext)))

        if not video_files:
            print(f"No video files found in {args.input}")
            return

        print(f"Found {len(video_files)} video(s) to process")

        for video_path in video_files:
            video_name = os.path.splitext(os.path.basename(video_path))[0]
            output_dir = os.path.join(args.output, video_name)
            processor.process_video(video_path, output_dir)
    else:
        print(f"Error: {args.input} is not a valid file or directory")
        sys.exit(1)


if __name__ == "__main__":
    main()
