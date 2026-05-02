"""
SAM-based (Segment Anything Model) segmentation module.

Given a point prompt (u, v) and an RGB image, returns a binary mask
for the object at that pixel location.

Uses facebook/sam-vit-base via HuggingFace transformers.
"""

from __future__ import annotations

import numpy as np
import torch
from PIL import Image
from transformers import SamModel, SamProcessor

import config

_model: SamModel | None = None
_processor: SamProcessor | None = None


def _load_model():
    global _model, _processor
    if _model is None:
        print(f"[SAM] Loading {config.SAM_MODEL} …")
        _processor = SamProcessor.from_pretrained(config.SAM_MODEL)
        _model = SamModel.from_pretrained(config.SAM_MODEL)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _model = _model.to(device).eval()
        print(f"[SAM] Model loaded on {device}")
    return _model, _processor


def get_mask(
    image_rgb: np.ndarray,
    prompt_point_uv: tuple[float, float],
    prompt_label: int = 1,  # 1=foreground
) -> tuple[np.ndarray, float] | tuple[None, float]:
    """
    Run SAM with a single point prompt.

    Args:
        image_rgb:        (H, W, 3) uint8 RGB image
        prompt_point_uv:  (u, v) pixel coordinates of the prompt point
        prompt_label:     1=foreground, 0=background

    Returns:
        (mask, score): binary mask (H, W bool), IoU score
        or (None, 0.0) on failure
    """
    model, processor = _load_model()
    device = next(model.parameters()).device

    pil_img = Image.fromarray(image_rgb.astype(np.uint8))
    u, v = float(prompt_point_uv[0]), float(prompt_point_uv[1])

    # SAM processor expects input_points as [[[x, y]]] (batch, num_points, 2)
    inputs = processor(
        images=pil_img,
        input_points=[[[u, v]]],
        input_labels=[[prompt_label]],
        return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    # Post-process masks back to original image resolution
    masks = processor.image_processor.post_process_masks(
        outputs.pred_masks.cpu(),
        inputs["original_sizes"].cpu(),
        inputs["reshaped_input_sizes"].cpu(),
    )
    # masks[0]: (1, 3, H, W)   — batch 0, object 0, 3 candidate masks
    # iou_scores: (1, 1, 3)
    iou_scores = outputs.iou_scores[0, 0].cpu().numpy()  # (3,)
    best_idx = int(iou_scores.argmax())
    best_mask = masks[0][0][best_idx].numpy().astype(bool)  # (H, W)
    best_score = float(iou_scores[best_idx])

    return best_mask, best_score


def get_mask_with_box(
    image_rgb: np.ndarray,
    box_xyxy: list[float],
) -> tuple[np.ndarray, float] | tuple[None, float]:
    """
    Run SAM with a bounding-box prompt.

    Args:
        image_rgb: (H, W, 3) uint8 RGB image
        box_xyxy:  [x1, y1, x2, y2] bounding box

    Returns:
        (mask, score) or (None, 0.0)
    """
    model, processor = _load_model()
    device = next(model.parameters()).device

    pil_img = Image.fromarray(image_rgb.astype(np.uint8))
    # input_boxes: [[[x1, y1, x2, y2]]]  (batch, num_boxes, 4)
    inputs = processor(
        images=pil_img,
        input_boxes=[[[box_xyxy[0], box_xyxy[1], box_xyxy[2], box_xyxy[3]]]],
        return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    masks = processor.image_processor.post_process_masks(
        outputs.pred_masks.cpu(),
        inputs["original_sizes"].cpu(),
        inputs["reshaped_input_sizes"].cpu(),
    )
    iou_scores = outputs.iou_scores[0, 0].cpu().numpy()
    best_idx = int(iou_scores.argmax())
    best_mask = masks[0][0][best_idx].numpy().astype(bool)
    best_score = float(iou_scores[best_idx])

    return best_mask, best_score


def mask_stats(
    mask: np.ndarray,
) -> dict:
    """
    Compute useful statistics of a binary mask for OBB estimation.

    Returns dict with:
        centroid_uv:  (u, v) centroid in pixel coordinates
        bbox_xyxy:    [x1, y1, x2, y2] tight bounding box
        area_px:      number of True pixels
        width_px:     bounding box width
        height_px:    bounding box height
    """
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return {
            "centroid_uv": None, "bbox_xyxy": None,
            "area_px": 0, "width_px": 0, "height_px": 0,
        }
    x1, x2 = int(xs.min()), int(xs.max())
    y1, y2 = int(ys.min()), int(ys.max())
    return {
        "centroid_uv": (float(xs.mean()), float(ys.mean())),
        "bbox_xyxy":   [x1, y1, x2, y2],
        "area_px":     int(len(xs)),
        "width_px":    x2 - x1,
        "height_px":   y2 - y1,
    }
