import numpy as np

def is_super_long_or_wide(mask, label, ratio=0.7):
    # Find foreground pixels of the specified label
    ys, xs = np.where(mask == label)
    if len(xs) == 0:
        return False  # No foreground found
    
    # Image dimensions
    h, w = mask.shape
    
    # Bounding box of the foreground object
    min_x, max_x = xs.min(), xs.max()
    min_y, max_y = ys.min(), ys.max()
    
    bbox_w = max_x - min_x + 1
    bbox_h = max_y - min_y + 1
    
    # Check if width or height of bounding box is large enough
    # relative to the entire image (e.g., ≥ 90%)
    return (bbox_w / w >= ratio) or (bbox_h / h >= ratio)

from PIL import Image

def resize_mask_with_unique_label(mask, target_h, target_w, label):
    """
    Resize a binary/segmentation mask to (target_h, target_w)
    ensuring only the given 'label' exists in the output mask.
    Nearest-neighbor used to avoid mixing labels.
    """

    # Convert to PIL for nearest-neighbor resize (no interpolation blending)
    mask_img = Image.fromarray((mask == label).astype(np.uint8))

    # Resize
    mask_resized = mask_img.resize((target_w, target_h), Image.NEAREST)

    # Convert back and fill only with the given label
    mask_out = np.array(mask_resized, dtype=np.uint8)
    mask_out = (mask_out > 0).astype(np.uint8) * label  # ensure only unique label

    return mask_out

import cv2

def keep_largest_component(mask: np.ndarray) -> np.ndarray:
    """
    Keep only the largest connected foreground component in a binary mask.
    Input:
        mask: HxW array, values 0 (background) / 1 (foreground)
    Output:
        same shape, only largest component kept as 1, others set to 0
    """
    mask_uint8 = (mask > 0).astype(np.uint8)
    num_labels, labels = cv2.connectedComponents(mask_uint8)

    # no foreground
    if num_labels <= 1:
        return np.zeros_like(mask)

    # count each label (skip background 0)
    counts = np.bincount(labels.ravel())
    counts[0] = 0
    largest_label = counts.argmax()

    largest_mask = (labels == largest_label).astype(mask.dtype)
    return largest_mask

