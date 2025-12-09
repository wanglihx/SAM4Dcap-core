import os

ROOT = os.path.dirname(__file__)

import argparse
import cv2
from PIL import Image
import yaml
from types import SimpleNamespace
import numpy as np
import torch.nn.functional as F

import random
import glob
from tqdm import tqdm

import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, 'models', 'sam_3d_body'))
sys.path.append(os.path.join(current_dir, 'models', 'diffusion_vas'))
# sys.path.append(parent_dir)

from utils import draw_point_marker, mask_painter, images_to_mp4, DAVIS_PALETTE, jpg_folder_to_mp4, is_super_long_or_wide, keep_largest_component, is_skinny_mask, bbox_from_mask, are_bboxes_similar

from models.sam_3d_body.sam_3d_body import load_sam_3d_body, SAM3DBodyEstimator
from models.sam_3d_body.notebook.utils import process_image_with_mask
from models.sam_3d_body.tools.vis_utils import visualize_sample_together, visualize_sample
from models.diffusion_vas.demo import init_amodal_segmentation_model, init_rgb_model, init_depth_model, load_and_transform_masks, load_and_transform_rgbs, rgb_to_depth

import torch
# select the device for computation
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"using device: {device}")
if device.type == "cuda":
    # use bfloat16 for the entire notebook
    # torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
    # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
elif device.type == "mps":
    print(
        "\nSupport for MPS devices is preliminary. SAM 3 is trained with CUDA and might "
        "give numerically different outputs and sometimes degraded performance on MPS. "
        "See e.g. https://github.com/pytorch/pytorch/issues/84936 for a discussion."
    )


import uuid
from datetime import datetime

def gen_id():
    t = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]  # 到毫秒
    u = uuid.uuid4().hex[:8]  # 取 8 位即可
    return f"{t}_{u}"


# ===============================
# Global runtime objects
# ===============================
CONFIG = None           # loaded config
sam3_model = None       # your SAM-3 model instance
sam3_image_model = None
image_predictor = None
predictor = None        # sam-3 predictor
inference_state = None  # sam-3 inference_state
RUNTIME = {}            # global dict to store runtime data per video/run


def load_config(config_path: str):
    """Load YAML config into a SimpleNamespace for convenient attribute access."""
    with open(config_path, "r") as f:
        cfg_dict = yaml.safe_load(f)
    return SimpleNamespace(**cfg_dict)


def build_sam3_3d_body_config(cfg):
    mhr_path = cfg.sam_3d_body['mhr_path']
    fov_path = cfg.sam_3d_body['fov_path']
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model, model_cfg = load_sam_3d_body(
        cfg.sam_3d_body['ckpt_path'], device=device, mhr_path=mhr_path
    )
    
    human_detector, human_segmentor, fov_estimator = None, None, None
    from models.sam_3d_body.tools.build_fov_estimator import FOVEstimator
    fov_estimator = FOVEstimator(name='moge2', device=device, path=fov_path)

    estimator = SAM3DBodyEstimator(
        sam_3d_body_model=model,
        model_cfg=model_cfg,
        human_detector=human_detector,
        human_segmentor=human_segmentor,
        fov_estimator=fov_estimator,
    )

    return estimator


def build_diffusion_vas_config(cfg):
    model_path_mask = cfg.completion['model_path_mask']
    model_path_rgb = cfg.completion['model_path_rgb']
    depth_encoder = cfg.completion['depth_encoder']
    model_path_depth = cfg.completion['model_path_depth']
    max_occ_len = min(cfg.completion['max_occ_len'], cfg.sam_3d_body['batch_size'])

    generator = torch.manual_seed(23)

    pipeline_mask = init_amodal_segmentation_model(model_path_mask)
    pipeline_rgb = init_rgb_model(model_path_rgb)
    model_path_depth = model_path_depth + f"/depth_anything_v2_{depth_encoder}.pth"
    depth_model = init_depth_model(model_path_depth, depth_encoder)

    return pipeline_mask, pipeline_rgb, depth_model, max_occ_len, generator


def init_runtime(config_path: str = os.path.join(ROOT, "configs", "body4d.yaml"), output_dir: str=''):
    """Initialize CONFIG, SAM3_MODEL, and global RUNTIME dict."""
    global CONFIG, sam3_3d_body_model, RUNTIME, OUTPUT_DIR, pipeline_mask \
        , pipeline_rgb, depth_model, max_occ_len, generator
    CONFIG = load_config(config_path)
    sam3_3d_body_model = build_sam3_3d_body_config(CONFIG)

    if CONFIG.completion.get('enable', False):
        pipeline_mask, pipeline_rgb, depth_model, max_occ_len, generator = build_diffusion_vas_config(CONFIG)
    else:
        pipeline_mask, pipeline_rgb, depth_model, max_occ_len, generator = None, None, None, None, None
    
    OUTPUT_DIR = output_dir

    RUNTIME = {}  # clear any old state
    RUNTIME['batch_size'] = CONFIG.sam_3d_body.get('batch_size', 1)
    RUNTIME['detection_resolution'] = CONFIG.completion.get('detection_resolution', [256, 512])
    RUNTIME['completion_resolution'] = CONFIG.completion.get('completion_resolution', [512, 1024])


def on_4d_generation(video_path: str=None):
    """
    Placeholder for 4D generation.
    Later:
      - run sam3_3d_body_model on per-frame images + masks
      - render 4D visualization video
    For now, just log and return None.
    """
    print("[DEBUG] 4D Generation button clicked.")
    # TODO: implement 4D body generation, write to a video and return its path

    IMAGE_PATH = os.path.join(OUTPUT_DIR, 'images') # for sam3-3d-body
    MASKS_PATH = os.path.join(OUTPUT_DIR, 'masks')  # for sam3-3d-body
    image_extensions = [
        "*.jpg",
        "*.jpeg",
        "*.png",
        "*.gif",
        "*.bmp",
        "*.tiff",
        "*.webp",
    ]
    images_list = sorted(
        [
            image
            for ext in image_extensions
            for image in glob.glob(os.path.join(IMAGE_PATH, ext))
        ]
    )
    masks_list = sorted(
        [
            image
            for ext in image_extensions
            for image in glob.glob(os.path.join(MASKS_PATH, ext))
        ]
    )

    # os.makedirs(f"{OUTPUT_DIR}/mask_4d", exist_ok=True)
    # os.makedirs(f"{OUTPUT_DIR}/mask_4d_individual", exist_ok=True)
    # batch_size = RUNTIME['batch_size']

    from PIL import Image
    import numpy as np

    img = Image.open(masks_list[0])
    mask_np = np.array(img)
    labels = np.unique(mask_np[mask_np != 0])
    RUNTIME['out_obj_ids'] = labels.tolist()

    os.makedirs(f"{OUTPUT_DIR}/mask_4d", exist_ok=True)
    os.makedirs(f"{OUTPUT_DIR}/mask_4d_individual", exist_ok=True)
    batch_size = RUNTIME['batch_size']
    n = len(images_list)
    
    # Optional, detect occlusions
    pred_res = RUNTIME['detection_resolution']
    pred_res_hi = RUNTIME['completion_resolution']
    modal_pixels_list = []
    if pipeline_mask is not None:
        for obj_id in RUNTIME['out_obj_ids']:
            modal_pixels, ori_shape = load_and_transform_masks(OUTPUT_DIR + "/masks", resolution=pred_res, obj_id=obj_id)
            modal_pixels_list.append(modal_pixels)
        rgb_pixels, _, raw_rgb_pixels = load_and_transform_rgbs(OUTPUT_DIR + "/images", resolution=pred_res)
        depth_pixels = rgb_to_depth(rgb_pixels, depth_model)

    mhr_shape_scale_dict = {}   # each element is a list storing input parameters for mhr_forward

    for i in tqdm(range(0, n, batch_size)):
        if i == 0:
            continue
        batch_images = images_list[i:i + batch_size]
        batch_masks  = masks_list[i:i + batch_size]

        W, H = Image.open(batch_masks[0]).size

        # Optional, detect occlusions
        idx_dict = {}
        idx_path = {}
        if len(modal_pixels_list) > 0:
            print("detect occlusions ...")
            for (modal_pixels, obj_id) in zip(modal_pixels_list, RUNTIME['out_obj_ids']):
                # detect occlusions for each object
                # predict amodal masks (amodal segmentation)
                pred_amodal_masks = pipeline_mask(
                    modal_pixels[:, i:i + batch_size, :, :, :],
                    depth_pixels[:, i:i + batch_size, :, :, :],
                    height=pred_res[0],
                    width=pred_res[1],
                    num_frames=modal_pixels[:, i:i + batch_size, :, :, :].shape[1],
                    decode_chunk_size=8,
                    motion_bucket_id=127,
                    fps=8,
                    noise_aug_strength=0.02,
                    min_guidance_scale=1.5,
                    max_guidance_scale=1.5,
                    generator=generator,
                ).frames[0]

                # for iou (localise occlusions spatial-temporally)
                pred_amodal_masks = [np.array(img.resize((W, H))) for img in pred_amodal_masks]
                pred_amodal_masks = np.array(pred_amodal_masks).astype('uint8')
                pred_amodal_masks = (pred_amodal_masks.sum(axis=-1) > 600).astype('uint8')
                pred_amodal_masks = [keep_largest_component(pamc) for pamc in pred_amodal_masks]    # avoid small noisy masks

                # compute iou
                ious = []
                obj_track_list = []
                masks = [(np.array(Image.open(bm).convert('P'))==obj_id).astype('uint8') for bm in batch_masks]
                for a, b in zip(masks, pred_amodal_masks):
                    area_a = (a > 0).sum()
                    area_b = (b > 0).sum()
                    if area_a == 0 and area_b == 0:
                        ious.append(1.0)
                    elif area_a > area_b:
                        ious.append(1.0)
                    else:
                        inter = np.logical_and(a > 0, b > 0).sum()
                        uni = np.logical_or(a > 0, b > 0).sum()
                        obj_iou = inter / (uni + 1e-6)
                        ious.append(obj_iou)

                        obj_track_list.append(bbox_from_mask(a))

                # remove fake completions (empty or from MARGINs)
                for pi, pamc in enumerate(pred_amodal_masks):
                    # zero predictions, back to original masks
                    if masks[pi].sum() > pred_amodal_masks[pi].sum():
                        ious[pi] = 1.0
                    elif len(obj_track_list)>0 and not are_bboxes_similar(bbox_from_mask(pamc), obj_track_list[-1]):
                        ious[pi] = 1.0
                    elif is_super_long_or_wide(pamc, obj_id):
                        ious[pi] = 1.0
                    elif is_skinny_mask(pamc):
                        ious[pi] = 1.0

                # confirm occlusions & save masks (for HMR)
                start, end = (idxs := [ix for ix,x in enumerate(ious) if x < 0.7]) and (idxs[0], idxs[-1]) or (None, None)
                if start is not None and end is not None:
                    start = max(0, start-2)
                    end = min(modal_pixels[:, i:i + batch_size, :, :, :].shape[1]-1, end+2)
                    idx_dict[obj_id] = (start, end)
                    completion_path = ''.join(random.choices('0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ', k=4))
                    completion_image_path = f'{OUTPUT_DIR}/completion/{completion_path}/images'
                    completion_masks_path = f'{OUTPUT_DIR}/completion/{completion_path}/masks'
                    os.makedirs(completion_image_path, exist_ok=True)
                    os.makedirs(completion_masks_path, exist_ok=True)
                    idx_path[obj_id] = {'images': completion_image_path, 'masks': completion_masks_path}

            # completion
            for obj_id, (start, end) in idx_dict.items(): 

                modal_pixels_current, ori_shape = load_and_transform_masks(OUTPUT_DIR + "/masks", resolution=pred_res_hi, obj_id=obj_id)
                modal_pixels_current = modal_pixels_current[:, i:i + batch_size, :, :, :][:, start:end]
                rgb_pixels_current, _, raw_rgb_pixels_current = load_and_transform_rgbs(OUTPUT_DIR + "/images", resolution=pred_res_hi)
                rgb_pixels_current = rgb_pixels_current[:, i:i + batch_size, :, :, :][:, start:end]
                depth_pixels_current = rgb_to_depth(rgb_pixels_current, depth_model)

                # mask re-completion (with higher resolution)
                pred_amodal_masks = pipeline_mask(
                    modal_pixels_current,
                    depth_pixels_current,
                    height=pred_res_hi[0],
                    width=pred_res_hi[1],
                    num_frames=end-start,
                    decode_chunk_size=8,
                    motion_bucket_id=127,
                    fps=8,
                    noise_aug_strength=0.02,
                    min_guidance_scale=1.5,
                    max_guidance_scale=1.5,
                    generator=generator,
                ).frames[0]

                pred_amodal_masks = np.array([np.array(img).astype('uint8') for img in pred_amodal_masks])
                pred_amodal_masks = (pred_amodal_masks.sum(axis=-1) > 600).astype('uint8')
                pred_amodal_masks = [keep_largest_component(pamc) for pamc in pred_amodal_masks]    # avoid small noisy masks

                # save completion masks
                completion_masks_path = idx_path[obj_id]['masks']
                for idx_ in range(end-start):
                    mask_idx_ = (pred_amodal_masks[idx_] > 0).astype(np.uint8)
                    mask_idx_ = Image.fromarray(mask_idx_, 'L').resize((ori_shape[1], ori_shape[0]), Image.NEAREST)
                    mask_idx_ = (np.array(mask_idx_) > 0).astype(np.uint8) * obj_id 
                    mask_idx_ = Image.fromarray(mask_idx_.astype(np.uint8), 'P')
                    mask_idx_.putpalette(DAVIS_PALETTE)
                    mask_idx_.save(os.path.join(completion_masks_path, f"{(idx_+start):08d}.png"))

                # prepare inputs
                completion_image_path = idx_path[obj_id]['images']
                pred_amodal_masks_current = pred_amodal_masks
                modal_mask_union = (modal_pixels_current[0, :, 0, :, :].cpu().numpy() > 0).astype('uint8')
                pred_amodal_masks_current = np.logical_or(pred_amodal_masks_current, modal_mask_union).astype('uint8')
                pred_amodal_masks_tensor = torch.from_numpy(np.where(pred_amodal_masks_current == 0, -1, 1)).float().unsqueeze(0).unsqueeze(
                    2).repeat(1, 1, 3, 1, 1)

                modal_obj_mask = (modal_pixels_current > 0).float()
                modal_background = 1 - modal_obj_mask
                rgb_pixels_current = (rgb_pixels_current + 1) / 2
                modal_rgb_pixels = rgb_pixels_current * modal_obj_mask + modal_background
                modal_rgb_pixels = modal_rgb_pixels * 2 - 1

                print("content completion by diffusion-vas ...")
                # predict amodal rgb (content completion)
                pred_amodal_rgb = pipeline_rgb(
                    modal_rgb_pixels,
                    pred_amodal_masks_tensor,
                    height=pred_res_hi[0], # my_res[0]
                    width=pred_res_hi[1],  # my_res[1]
                    num_frames=end-start,
                    decode_chunk_size=8,
                    motion_bucket_id=127,
                    fps=8,
                    noise_aug_strength=0.02,
                    min_guidance_scale=1.5,
                    max_guidance_scale=1.5,
                    generator=generator,
                ).frames[0]

                pred_amodal_rgb = [np.array(img) for img in pred_amodal_rgb]

                # save pred_amodal_rgb
                pred_amodal_rgb = np.array(pred_amodal_rgb).astype('uint8')
                pred_amodal_rgb_save = np.array([cv2.resize(frame, (ori_shape[1], ori_shape[0]), interpolation=cv2.INTER_LINEAR)
                                                for frame in pred_amodal_rgb])
                idx_ = start
                for img in pred_amodal_rgb_save:
                    cv2.imwrite(os.path.join(completion_image_path, f"{idx_:08d}.jpg"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
                    idx_ += 1

        # Process with external mask
        mask_outputs, id_batch = process_image_with_mask(sam3_3d_body_model, batch_images, batch_masks, idx_path, idx_dict, mhr_shape_scale_dict)
        
        for image_path, mask_output, id_current in zip(batch_images, mask_outputs, id_batch):
            img = cv2.imread(image_path)
            rend_img = visualize_sample_together(img, mask_output, sam3_3d_body_model.faces, id_current)
            cv2.imwrite(
                f"{OUTPUT_DIR}/mask_4d/{os.path.basename(image_path)[:-4]}.jpg",
                rend_img.astype(np.uint8),
            )
            rend_img_list = visualize_sample(img, mask_output, sam3_3d_body_model.faces, id_current)
            for ri, rend_img in enumerate(rend_img_list):
                cv2.imwrite(
                    f"{OUTPUT_DIR}/mask_4d_individual/{os.path.basename(image_path)[:-4]}_{ri+1}.jpg",
                    rend_img.astype(np.uint8),
                )

    jpg_folder_to_mp4(f"{OUTPUT_DIR}/mask_4d", f"{OUTPUT_DIR}/4d.mp4", fps=25)

    return f"{OUTPUT_DIR}/4d.mp4"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Offline 4D Body Generation for long videos")
    parser.add_argument("--output_dir", type=str, default="path to outputs/20251207_043551_865_21ed56bf",
                        help="Path to the output directory")
    args = parser.parse_args()
    # Check dir
    if not os.path.isdir(args.output_dir):
        raise FileNotFoundError(f"Output directory not found: {args.output_dir}")

    init_runtime(output_dir=args.output_dir)
    on_4d_generation()
