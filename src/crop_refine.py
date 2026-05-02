"""
Crop-and-refine: for small connectors, crop a tight window around the
projected prior center, run SAM on the crop for accurate segmentation,
then map back to full-image coordinates.

This avoids SAM grabbing large background structures when processing
full 2560×1440 images for ~30px wide connectors.
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import config
import utils
import sam_segment


def crop_around_point(
    image: np.ndarray,
    u: float,
    v: float,
    crop_r: int = 200,
) -> tuple[np.ndarray, int, int]:
    """
    Crop a (2*crop_r)×(2*crop_r) patch around (u, v).
    Returns (crop, x_offset, y_offset) where offsets map crop coords → full image.
    """
    H, W = image.shape[:2]
    x0 = max(0, int(u) - crop_r)
    y0 = max(0, int(v) - crop_r)
    x1 = min(W, int(u) + crop_r)
    y1 = min(H, int(v) + crop_r)
    return image[y0:y1, x0:x1].copy(), x0, y0


def sam_on_crop(
    image_rgb: np.ndarray,
    u: float,
    v: float,
    crop_r: int = 200,
    min_score: float = 0.70,
    min_area_px: int = 30,
) -> dict | None:
    """
    Crop around (u,v), run SAM with point prompt at the crop center,
    then map the result back to full-image coordinates.

    Returns dict with full-image centroid, bbox_xyxy, area, mask, score
    or None if failed.
    """
    crop, x0, y0 = crop_around_point(image_rgb, u, v, crop_r)
    if crop.size == 0:
        return None

    # Prompt at center of crop (where the port should be)
    crop_u = u - x0
    crop_v = v - y0

    mask_crop, score = sam_segment.get_mask(crop, (crop_u, crop_v))
    stats_crop = sam_segment.mask_stats(mask_crop)

    # Also try box prompt (crop bbox = whole crop center ± physical size)
    if score < min_score or stats_crop["area_px"] < min_area_px:
        # Box covering central 60% of the crop
        bw, bh = crop.shape[1], crop.shape[0]
        box_crop = [bw * 0.1, bh * 0.1, bw * 0.9, bh * 0.9]
        mask2, score2 = sam_segment.get_mask_with_box(crop, box_crop)
        stats2 = sam_segment.mask_stats(mask2)
        if score2 > score and stats2["area_px"] > stats_crop["area_px"]:
            mask_crop, score, stats_crop = mask2, score2, stats2

    if score < min_score or stats_crop["area_px"] < min_area_px:
        return None
    if stats_crop["centroid_uv"] is None:
        return None

    # Check that the mask doesn't cover most of the crop (would mean SAM
    # grabbed a large background region rather than the small connector)
    crop_area = crop.shape[0] * crop.shape[1]
    mask_fraction = stats_crop["area_px"] / crop_area
    if mask_fraction > 0.5:
        return None   # mask too large relative to crop → probably background

    # Map centroid and bbox back to full-image coords
    cx_crop, cy_crop = stats_crop["centroid_uv"]
    cx_full = cx_crop + x0
    cy_full = cy_crop + y0
    x1c, y1c, x2c, y2c = stats_crop["bbox_xyxy"]
    bbox_full = [x1c + x0, y1c + y0, x2c + x0, y2c + y0]

    # Reconstruct full-image mask
    H, W = image_rgb.shape[:2]
    mask_full = np.zeros((H, W), dtype=bool)
    ys_c, xs_c = np.where(mask_crop)
    ys_f = ys_c + y0
    xs_f = xs_c + x0
    valid = (ys_f < H) & (xs_f < W)
    mask_full[ys_f[valid], xs_f[valid]] = True

    return {
        "centroid_uv": (cx_full, cy_full),
        "bbox_xyxy":   bbox_full,
        "width_px":    x2c - x1c,
        "height_px":   y2c - y1c,
        "area_px":     stats_crop["area_px"],
        "mask_fraction": mask_fraction,
        "sam_score":   score,
        "mask_full":   mask_full,
    }


def collect_crop_sam_observations(
    entity_name: str,
    prior_center: np.ndarray,
    frames: list[int],
    poses: dict,
    K: np.ndarray,
    crop_r: int = 200,
    min_sam_score: float = 0.70,
    max_centroid_dist_px: float = 150,
) -> dict:
    """
    Collect SAM observations using the crop-and-refine approach.
    Only processes frames where the prior projects cleanly into the image.
    """
    print(f"\n  Crop-SAM for {entity_name} (crop_r={crop_r}px) …")
    obs = {
        "centroids_uv": [],
        "bbox_xyxy":    [],
        "c2ws":         [],
        "frame_nums":   [],
        "depths":       [],
        "sam_scores":   [],
        "widths_px":    [],
        "heights_px":   [],
    }

    for frame_num in frames:
        c2w = poses[frame_num]
        if not utils.is_visible(prior_center, c2w, K, margin=crop_r + 50):
            continue
        pix, depth = utils.project_to_image(prior_center, c2w, K)
        u, v = float(pix[0]), float(pix[1])

        img_rgb = utils.load_image_rgb(frame_num)
        result = sam_on_crop(img_rgb, u, v, crop_r=crop_r,
                             min_score=min_sam_score)
        if result is None:
            print(f"    [{frame_num}] crop-SAM failed")
            continue

        cx, cy = result["centroid_uv"]
        dist = np.sqrt((cx - u) ** 2 + (cy - v) ** 2)
        if dist > max_centroid_dist_px:
            print(f"    [{frame_num}] centroid {dist:.0f}px from prior — skip")
            continue

        print(f"    [{frame_num}] score={result['sam_score']:.3f}  "
              f"centroid=({cx:.0f},{cy:.0f}) prior=({u:.0f},{v:.0f}) "
              f"dist={dist:.0f}px  "
              f"w={result['width_px']}px h={result['height_px']}px  "
              f"fill={result['mask_fraction']:.2f}")

        obs["centroids_uv"].append((cx, cy))
        obs["bbox_xyxy"].append(result["bbox_xyxy"])
        obs["c2ws"].append(c2w)
        obs["frame_nums"].append(frame_num)
        obs["depths"].append(depth)
        obs["sam_scores"].append(result["sam_score"])
        obs["widths_px"].append(result["width_px"])
        obs["heights_px"].append(result["height_px"])

    print(f"  → {len(obs['centroids_uv'])} valid crop-SAM observations")
    return obs


def save_crop_debug(
    entity_name: str,
    prior_center: np.ndarray,
    frames: list[int],
    poses: dict,
    K: np.ndarray,
    out_dir: str,
    crop_r: int = 250,
) -> None:
    """
    Save annotated crops for visual inspection of each frame.
    """
    import os
    os.makedirs(out_dir, exist_ok=True)

    for frame_num in frames:
        c2w = poses[frame_num]
        if not utils.is_visible(prior_center, c2w, K, margin=crop_r):
            continue
        pix, depth = utils.project_to_image(prior_center, c2w, K)
        u, v = float(pix[0]), float(pix[1])

        img_rgb = utils.load_image_rgb(frame_num)
        crop_rgb, x0, y0 = crop_around_point(img_rgb, u, v, crop_r)
        crop_bgr = crop_rgb[:, :, ::-1].copy()

        # Mark the projected prior center in the crop
        cu, cv = int(u - x0), int(v - y0)
        cv2.circle(crop_bgr, (cu, cv), 10, (0, 255, 255), 2)
        cv2.putText(crop_bgr, f"{entity_name} d={depth:.2f}m",
                    (5, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        # Scale up for easier viewing
        scale = 2.0
        h, w = crop_bgr.shape[:2]
        crop_bgr = cv2.resize(crop_bgr, (int(w * scale), int(h * scale)),
                              interpolation=cv2.INTER_LANCZOS4)

        out_path = f"{out_dir}/crop_{entity_name}_{frame_num:06d}.jpg"
        cv2.imwrite(out_path, crop_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])

    print(f"  Saved crops to {out_dir}")
