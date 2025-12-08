import cv2
import numpy as np

def concat_pngs_side_by_side(pngs_left, pngs_right, output_path, fps=30):
    """
    Concatenate two lists of PNG images side by side and save them as an MP4 video.

    Args:
        pngs_left (list[str]): List of PNG file paths for the left side
        pngs_right (list[str]): List of PNG file paths for the right side
        output_path (str): Output MP4 path
        fps (int): Frames per second
    """

    n = min(len(pngs_left), len(pngs_right))
    if n == 0:
        raise ValueError("Input PNG lists cannot be empty")

    # Read the first frame to determine resolution
    left0 = cv2.imread(pngs_left[0])
    right0 = cv2.imread(pngs_right[0])

    if left0 is None or right0 is None:
        raise ValueError("Failed to read PNG images")

    # If resolution differs, resize right to match left
    h, w, _ = left0.shape
    right0 = cv2.resize(right0, (w, h))

    # Output resolution after horizontal concatenation
    out_w = w * 2
    out_h = h

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (out_w, out_h))

    for i in range(n):
        left = cv2.imread(pngs_left[i])
        right = cv2.imread(pngs_right[i])

        if left is None or right is None:
            print(f"Warning: Failed to read frame {i}, skipped")
            continue

        # Ensure size alignment
        right = cv2.resize(right, (w, h))

        # Concatenate horizontally
        concat = np.concatenate([left, right], axis=1)

        writer.write(concat)

    writer.release()
    print(f"Video saved to: {output_path}")


if __name__ == "__main__":
    import os, glob
    root = "path to image folder"
    png_list = glob.glob(os.path.join(root, '*'))
    png_list.sort()

    png1 = [os.path.join(p, 'mask_' + p.split('_')[-1] + '_bbox_000.png') for p in png_list]
    png2 = [os.path.join(p, 'mask_' + p.split('_')[-1] + '_overlay_000.png') for p in png_list]

    concat_pngs_side_by_side(png1, png2, 'mini.mp4', fps=30)
