import argparse
import os, sys, glob
from tqdm import tqdm
from PIL import Image
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(current_dir))

# from offline_app import inference configs
from offline_app import *


def inference(args):
    # init configs and cover with cmd options
    predictor = offline_app(refine_occlusion=args.refine_occlusion)

    # init data
    test_seq_name_list = glob.glob(os.path.join(args.data_dir, 'sequenceFiles', 'test', '*'))
    test_seq_name_list.sort()
    test_seq_name_list = [os.path.splitext(os.path.basename(tn))[0] for tn in test_seq_name_list]
    bboxes = torch.load(os.path.join(args.data_dir, 'body4d_3dpw_bbx_xyxy_uint16.pt'))

    batch_size = 200

    # inference
    for seq in tqdm(test_seq_name_list):
        # 0. init outputs
        output_dir = os.path.join(args.output_dir, seq)
        predictor.OUTPUT_DIR = output_dir
        os.makedirs(predictor.OUTPUT_DIR, exist_ok=True)
        frame_list = glob.glob(os.path.join(args.data_dir, 'imageFiles', seq, '*.jpg'))
        frame_list.sort()
        one_frame = Image.open(frame_list[0]).convert('RGB')
        width, height = one_frame.size

        for i in range(0, len(frame_list), batch_size):
            batch_frames = frame_list[i:i + batch_size]
            inference_state = predictor.predictor.init_state(video_path=batch_frames)
            predictor.predictor.clear_all_points_in_video(inference_state)
            predictor.RUNTIME['inference_state'] = inference_state
            predictor.RUNTIME['out_obj_ids'] = []

            ann_frame_idx = i

            # 1. load bbox (first frame)
            for obj_id in range(3):
                seq_name_with_id = f'{seq}_{obj_id}'
                try:
                    # only consider the first frame bbox
                    bbox = bboxes[seq_name_with_id]['bbx_xyxy'][ann_frame_idx].numpy()
                    # Let's add a box at (x_min, y_min, x_max, y_max) = (300, 0, 500, 400) to get started
                    box = np.array([bbox], dtype=np.float32)
                    rel_box = [[xmin / width, ymin / height, xmax / width, ymax / height] for xmin, ymin, xmax, ymax in box]
                    rel_box = np.array(rel_box, dtype=np.float32)
                    _, predictor.RUNTIME['out_obj_ids'], low_res_masks, video_res_masks = predictor.predictor.add_new_points_or_box(
                        inference_state=predictor.RUNTIME['inference_state'],
                        frame_idx=0,
                        obj_id=obj_id+1,
                        box=rel_box,
                    )
                except:
                    break

            # 3. tracking
            predictor.on_mask_generation(start_frame_idx=i)
        # 4. hmr upon masks

        with torch.autocast("cuda", enabled=False):
            predictor.on_4d_generation()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference on 3DPW")
    parser.add_argument("--data_dir", type=str, default="path to 3DPW data",
        help="Path to the data directory")
    parser.add_argument("--output_dir", type=str, default="path to output",
        help="Path to the output directory")
    parser.add_argument("--refine_occlusion", action="store_true",
        help="Whether to use occlusion-aware refinement (default False)")
    args = parser.parse_args()

    inference(args)
