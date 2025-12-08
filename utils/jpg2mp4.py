import os
import glob
import cv2

def jpg_folder_to_mp4(folder: str, output_filename: str, fps: int = 25):
    """
    Convert JPG images in a folder into an MP4 video, sorted by filename.

    Parameters
    ----------
    folder : str
        Path to the folder containing JPG images.
    output_filename : str
        Output MP4 file path (e.g., "output.mp4", can include full path).
    fps : int, default 25
        Video frame rate.
    """
    # Gather all JPG images (case-insensitive patterns)
    patterns = ["*.jpg", "*.JPG", "*.jpeg", "*.JPEG"]
    img_paths = []
    for p in patterns:
        img_paths.extend(glob.glob(os.path.join(folder, p)))

    if not img_paths:
        raise ValueError(f"No JPG images found in folder: {folder}")

    # Sort by filename
    img_paths = sorted(img_paths)

    # Read the first image to determine resolution
    first_img = cv2.imread(img_paths[0])
    if first_img is None:
        raise ValueError(f"Failed to read image: {img_paths[0]}")
    h, w = first_img.shape[:2]

    # Init video writer (mp4v codec)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_filename, fourcc, fps, (w, h))

    for path in img_paths:
        img = cv2.imread(path)
        if img is None:
            print(f"Warning: skipped unreadable image {path}")
            continue
        # Force resize if resolution differs from the first frame
        if img.shape[0] != h or img.shape[1] != w:
            img = cv2.resize(img, (w, h))
        writer.write(img)

    writer.release()
    print(f"Saved video to: {output_filename}")
