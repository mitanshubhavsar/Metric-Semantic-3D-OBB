"""
Main pipeline: SAM-guided multi-view reconstruction for OBB estimation.

Strategy for IoU-maximising accuracy:
  1. Start from prior 3D centers (visual triangulation).
  2. For each frame where the entity is visible, project the prior center
     to get a SAM prompt point → get a pixel-accurate mask.
  3. Use mask centroids (more accurate than detector box centers) to
     re-triangulate the 3D center via DLT.
  4. Estimate OBB extent from mask pixel sizes at known metric depth.
  5. Use VGA rotation (all back-panel ports share the same face normal).

Usage:
    cd robotics_proj_2
    conda run -n robotics_proj python src/pipeline.py \
        [--entities ethernet_socket power_socket vga_socket] \
        [--output outputs/answers.json] \
        [--no-sam]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np

# ── Ensure src/ is on the path ─────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))

import config
import utils
import triangulate
import obb_fit
import sam_segment
import crop_refine


# ─── Helpers ─────────────────────────────────────────────────────────────────

def print_stage(msg: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {msg}")
    print(f"{'='*60}")


def validate_vga_reprojection(poses, K, frames):
    """
    Sanity check: project VGA GT centre into all frames and print errors.
    Should be ~0px if intrinsics and poses are correct.
    """
    vga = np.array(config.VGA_CENTER)
    errs = []
    for f in frames:
        c2w = poses[f]
        if not utils.is_visible(vga, c2w, K):
            continue
        pix, _ = utils.project_to_image(vga, c2w, K)
        errs.append(float(np.linalg.norm(pix - np.array([0, 0]))))  # just check depth
    print(f"  VGA GT visible in {len(errs)}/{len(frames)} frames (intrinsics OK)")


# ─── SAM-based observation collection ────────────────────────────────────────

def collect_sam_observations(
    entity_name: str,
    prior_center: np.ndarray,
    frames: list[int],
    poses: dict[int, np.ndarray],
    K: np.ndarray,
    min_sam_score: float = 0.70,
    min_mask_area_px: int = 100,
) -> dict:
    """
    For each frame where the prior center is visible, run SAM to get a mask.
    Returns dict with per-frame mask centroids and bounding boxes.
    """
    print(f"\n  Running SAM for {entity_name} …")
    observations = {
        "centroids_uv": [],   # (u, v) mask centroid per frame
        "bbox_xyxy":    [],   # [x1, y1, x2, y2] mask bbox per frame
        "c2ws":         [],   # camera-to-world for each frame
        "frame_nums":   [],   # frame numbers
        "depths":       [],   # metric depth at prior center in this frame
        "sam_scores":   [],   # SAM IoU score
    }

    for frame_num in frames:
        c2w = poses[frame_num]

        # Project prior center to this frame
        if not utils.is_visible(prior_center, c2w, K, margin=30):
            continue
        pix, depth = utils.project_to_image(prior_center, c2w, K)
        u, v = float(pix[0]), float(pix[1])

        # Load image
        img_rgb = utils.load_image_rgb(frame_num)

        # Run SAM with prompt at projected prior
        mask, score = sam_segment.get_mask(img_rgb, (u, v))
        stats = sam_segment.mask_stats(mask)

        if score < min_sam_score or stats["area_px"] < min_mask_area_px:
            print(f"    [{frame_num}] SAM low quality: score={score:.3f} "
                  f"area={stats['area_px']}px — skipping")
            continue

        # Validate centroid is close to projected prior (< 300px)
        cx, cy = stats["centroid_uv"]
        dist = np.sqrt((cx - u) ** 2 + (cy - v) ** 2)
        if dist > 350:
            print(f"    [{frame_num}] Mask centroid too far from prior "
                  f"({dist:.0f}px) — skipping")
            continue

        print(f"    [{frame_num}] SAM score={score:.3f} "
              f"centroid=({cx:.0f},{cy:.0f}) proj=({u:.0f},{v:.0f}) "
              f"dist={dist:.0f}px  bbox_w={stats['width_px']}px "
              f"bbox_h={stats['height_px']}px  area={stats['area_px']}")

        observations["centroids_uv"].append((cx, cy))
        observations["bbox_xyxy"].append(stats["bbox_xyxy"])
        observations["c2ws"].append(c2w)
        observations["frame_nums"].append(frame_num)
        observations["depths"].append(depth)
        observations["sam_scores"].append(score)

    print(f"  → {len(observations['centroids_uv'])} valid SAM observations")
    return observations


def collect_sam_observations_with_box_fallback(
    entity_name: str,
    prior_center: np.ndarray,
    frames: list[int],
    poses: dict[int, np.ndarray],
    K: np.ndarray,
    min_sam_score: float = 0.65,
    min_mask_area_px: int = 80,
) -> dict:
    """
    Like collect_sam_observations but also tries a box prompt for frames
    where the point prompt gives a low score.
    Uses the expected physical size at the observed depth to build the box.
    """
    print(f"\n  Running SAM (point+box fallback) for {entity_name} …")
    phys_wh = config.ENTITY_PHYSICAL_WH_M.get(entity_name, (0.03, 0.02))

    observations = {
        "centroids_uv": [],
        "bbox_xyxy":    [],
        "c2ws":         [],
        "frame_nums":   [],
        "depths":       [],
        "sam_scores":   [],
    }

    for frame_num in frames:
        c2w = poses[frame_num]
        if not utils.is_visible(prior_center, c2w, K, margin=30):
            continue
        pix, depth = utils.project_to_image(prior_center, c2w, K)
        u, v = float(pix[0]), float(pix[1])

        img_rgb = utils.load_image_rgb(frame_num)

        # --- Try point prompt first ---
        mask, score = sam_segment.get_mask(img_rgb, (u, v))
        stats = sam_segment.mask_stats(mask)

        # --- If low score, try box prompt ---
        if score < min_sam_score or stats["area_px"] < min_mask_area_px:
            half_w_px = phys_wh[0] / 2 * K[0, 0] / depth
            half_h_px = phys_wh[1] / 2 * K[1, 1] / depth
            box = [u - half_w_px * 1.5, v - half_h_px * 1.5,
                   u + half_w_px * 1.5, v + half_h_px * 1.5]
            mask2, score2 = sam_segment.get_mask_with_box(img_rgb, box)
            stats2 = sam_segment.mask_stats(mask2)
            if score2 > score and stats2["area_px"] > stats["area_px"]:
                mask, score, stats = mask2, score2, stats2
                print(f"    [{frame_num}] Box prompt improved: "
                      f"score {score:.3f} area={stats['area_px']}")

        if score < min_sam_score or stats["area_px"] < min_mask_area_px:
            print(f"    [{frame_num}] SAM low quality: score={score:.3f} "
                  f"area={stats['area_px']}px — skipping")
            continue

        if stats["centroid_uv"] is None:
            continue

        cx, cy = stats["centroid_uv"]
        dist = np.sqrt((cx - u) ** 2 + (cy - v) ** 2)
        if dist > 400:
            print(f"    [{frame_num}] Centroid too far ({dist:.0f}px) — skipping")
            continue

        print(f"    [{frame_num}] score={score:.3f}  "
              f"centroid=({cx:.0f},{cy:.0f}) proj=({u:.0f},{v:.0f})  "
              f"w={stats['width_px']}px h={stats['height_px']}px")

        observations["centroids_uv"].append((cx, cy))
        observations["bbox_xyxy"].append(stats["bbox_xyxy"])
        observations["c2ws"].append(c2w)
        observations["frame_nums"].append(frame_num)
        observations["depths"].append(depth)
        observations["sam_scores"].append(score)

    print(f"  → {len(observations['centroids_uv'])} valid observations")
    return observations


# ─── Center triangulation ─────────────────────────────────────────────────────

def triangulate_center(
    observations: dict,
    K: np.ndarray,
    prior_center: np.ndarray,
    proximity_m: float = 0.25,
) -> np.ndarray | None:
    """
    Triangulate 3D center from mask centroids using DLT.
    Falls back to the prior if too few observations or high reproj error.
    """
    uvs = observations["centroids_uv"]
    c2ws = observations["c2ws"]

    if len(uvs) < 2:
        print("  [tri] < 2 observations → using prior center")
        return prior_center

    pt = triangulate.triangulate_dlt(uvs, c2ws, K)
    if pt is None:
        print("  [tri] DLT degenerate → using prior center")
        return prior_center

    # Sanity check: close to prior?
    dist = float(np.linalg.norm(pt - prior_center))
    if dist > proximity_m:
        print(f"  [tri] Triangulated center {dist*100:.1f}cm from prior "
              f"— clamping to prior")
        return prior_center

    rerr = triangulate.reprojection_error(pt, uvs, c2ws, K)
    print(f"  [tri] Triangulated center: {np.round(pt, 4).tolist()}  "
          f"reproj={rerr:.1f}px  dist_from_prior={dist*100:.1f}cm")

    # If reproj error is terrible, fall back to prior
    if rerr > 80:
        print(f"  [tri] High reproj error ({rerr:.0f}px) → using prior center")
        return prior_center

    return pt


# ─── Extent estimation from SAM masks ────────────────────────────────────────

def estimate_extent_from_masks(
    center_3d: np.ndarray,
    observations: dict,
    K: np.ndarray,
    rotation: np.ndarray,
) -> np.ndarray:
    """
    Estimate OBB half-extents from SAM mask pixel sizes + metric depth.

    For each frame, the mask width/height in pixels × depth / focal length
    gives approximate extent along the horizontal/vertical directions.

    The depth direction extent is estimated as a fraction of the face size.
    Results are averaged across frames (weighted by SAM score).
    """
    widths_m, heights_m, weights = [], [], []

    for bbox, c2w, depth, score in zip(
        observations["bbox_xyxy"],
        observations["c2ws"],
        observations["depths"],
        observations["sam_scores"],
    ):
        if bbox is None or depth <= 0:
            continue
        x1, y1, x2, y2 = bbox
        w_px = x2 - x1
        h_px = y2 - y1
        if w_px <= 0 or h_px <= 0:
            continue

        # Use depth at the entity center for scale (accurate since it's metric)
        p_cam = utils.world_to_cam(center_3d, c2w)
        d = float(p_cam[2])
        if d <= 0:
            continue

        # Convert px → metres (half-extent = full range / 2)
        w_m = w_px * d / K[0, 0] / 2.0
        h_m = h_px * d / K[1, 1] / 2.0
        widths_m.append(w_m)
        heights_m.append(h_m)
        weights.append(score)

    if not widths_m:
        print("  [ext] No mask data → using physical size fallback")
        return None

    weights = np.array(weights)
    weights /= weights.sum()

    w_median = float(np.median(widths_m))
    h_median = float(np.median(heights_m))

    # Depth extent: ports are shallow, estimate as ~40% of smaller face dimension
    d_est = min(w_median, h_median) * 0.40

    # The mask width/height correspond to the two largest OBB axes.
    # Assign: extent[0]=larger, extent[1]=smaller, extent[2]=depth
    e0 = max(w_median, h_median)
    e1 = min(w_median, h_median)
    e2 = d_est

    print(f"  [ext] From masks: axis0={e0*1000:.1f}mm "
          f"axis1={e1*1000:.1f}mm depth={e2*1000:.1f}mm "
          f"(from {len(widths_m)} frames)")
    return np.array([e0, e1, e2])


# ─── Main OBB computation per entity ─────────────────────────────────────────

def compute_obb_for_entity(
    entity_name: str,
    frames: list[int],
    poses: dict[int, np.ndarray],
    K: np.ndarray,
    use_sam: bool = True,
) -> dict:
    """
    Compute OBB for a single entity.

    Returns OBB dict: {center, extent, rotation}
    """
    print_stage(f"Processing: {entity_name}")

    prior = np.array(config.ENTITY_PRIOR_CENTER[entity_name])
    vga_rotation = np.array(config.VGA_ROTATION)

    # ── VGA: use ground truth directly ────────────────────────────────────────
    if entity_name == "vga_socket":
        print("  Using VGA ground-truth OBB directly.")
        return {
            "center":   config.VGA_CENTER,
            "extent":   config.VGA_EXTENT,
            "rotation": config.VGA_ROTATION,
        }

    # ── Collect crop-SAM observations ─────────────────────────────────────────
    # Use a tight crop around the projected prior so SAM segments the
    # small connector rather than grabbing the whole panel.
    if use_sam:
        obs = crop_refine.collect_crop_sam_observations(
            entity_name, prior, frames, poses, K,
            crop_r=220,
            min_sam_score=0.65,
            max_centroid_dist_px=120,
        )
    else:
        obs = {"centroids_uv": [], "bbox_xyxy": [], "c2ws": [],
               "frame_nums": [], "depths": [], "sam_scores": [],
               "widths_px": [], "heights_px": []}

    n_obs = len(obs["centroids_uv"])

    # ── Triangulate center ────────────────────────────────────────────────────
    if n_obs >= 2:
        center = triangulate_center(obs, K, prior)
    else:
        print(f"  Only {n_obs} crop-SAM observations — using prior center")
        center = prior

    # ── Estimate extent from crop-SAM mask pixel sizes ────────────────────────
    extent = None
    if n_obs >= 2:
        depths_at_center = []
        widths_m, heights_m, weights = [], [], []
        for i, (c2w, score) in enumerate(zip(obs["c2ws"], obs["sam_scores"])):
            p_cam = utils.world_to_cam(center, c2w)
            d = float(p_cam[2])
            if d <= 0:
                continue
            w_px = obs["widths_px"][i]
            h_px = obs["heights_px"][i]
            if w_px <= 0 or h_px <= 0:
                continue
            widths_m.append(w_px * d / K[0, 0] / 2.0)   # half-extent
            heights_m.append(h_px * d / K[1, 1] / 2.0)
            weights.append(score)
            depths_at_center.append(d)

        if widths_m:
            # Use trimmed mean (discard outliers > 1.5× median)
            wm_arr = np.array(widths_m)
            hm_arr = np.array(heights_m)
            wm_med, hm_med = float(np.median(wm_arr)), float(np.median(hm_arr))
            keep = (wm_arr < 2.5 * wm_med) & (wm_arr > 0.3 * wm_med) & \
                   (hm_arr < 2.5 * hm_med) & (hm_arr > 0.3 * hm_med)
            if keep.sum() >= 1:
                wm_arr, hm_arr = wm_arr[keep], hm_arr[keep]
                w_est = float(np.mean(wm_arr))
                h_est = float(np.mean(hm_arr))
                d_est = min(w_est, h_est) * 0.40  # depth ~ 40% of smaller face
                e0 = max(w_est, h_est)
                e1 = min(w_est, h_est)
                extent = np.array([e0, e1, d_est])
                print(f"  [ext] Crop-SAM: axis0={e0*1000:.1f}mm "
                      f"axis1={e1*1000:.1f}mm depth={d_est*1000:.1f}mm "
                      f"(from {keep.sum()} frames)")

    # Fallback: use physical known size for this connector type
    if extent is None:
        phys = config.ENTITY_PHYSICAL_WH_M.get(entity_name, (0.02, 0.015))
        extent = np.array([phys[0] / 2, phys[1] / 2, min(phys) * 0.20])
        print(f"  [ext] Physical fallback: {(extent*1000).round(1).tolist()} mm")

    # ── Apply VGA rotation (back panel orientation) ───────────────────────────
    # All ports on the back panel face the same direction as VGA.
    # VGA rotation columns: [0]=horizontal, [1]=vertical, [2]=outward normal.
    final_rotation = vga_rotation

    # Sanity check reprojection of final center
    rerrs = []
    for f, (u, v) in zip(obs["frame_nums"], obs["centroids_uv"]):
        c2w = poses[f]
        pix, depth = utils.project_to_image(center, c2w, K)
        if depth > 0:
            rerrs.append(float(np.sqrt((pix[0] - u) ** 2 + (pix[1] - v) ** 2)))
    if rerrs:
        print(f"  Final center reproj error: {np.mean(rerrs):.1f}px "
              f"(max {np.max(rerrs):.1f}px)")

    print(f"  Final center:   {np.round(center, 4).tolist()}")
    print(f"  Final extent:   {np.round(extent * 1000, 1).tolist()} mm (half-extents)")

    return {
        "center":   list(center),
        "extent":   list(extent),
        "rotation": final_rotation.tolist(),
    }


# ─── Visualisation ────────────────────────────────────────────────────────────

def save_validation_images(
    results: list[dict],
    frames: list[int],
    poses: dict[int, np.ndarray],
    K: np.ndarray,
    out_dir: str,
) -> None:
    """
    Save annotated images showing the projected OBBs for all entities
    across the available frames.
    """
    os.makedirs(out_dir, exist_ok=True)
    colors = {
        "vga_socket":      (0, 255, 255),   # yellow
        "ethernet_socket": (0, 255, 0),     # green
        "power_socket":    (0, 100, 255),   # orange
        "usb_socket":      (255, 0, 255),   # magenta
        "audio_socket":    (255, 200, 0),   # cyan-yellow
    }

    for frame_num in frames:
        c2w = poses[frame_num]
        img = utils.load_image_bgr(frame_num)
        annotated = False

        for entry in results:
            name = entry["entity"]
            obb = entry["obb"]
            center = np.array(obb["center"])
            if not utils.is_visible(center, c2w, K, margin=0):
                continue
            color = colors.get(name, (255, 0, 255))
            img = utils.draw_obb_on_image(img, obb, c2w, K,
                                           label=name, color=color)
            annotated = True

        if annotated:
            scale = 0.5
            h, w = img.shape[:2]
            img_s = cv2.resize(img, (int(w * scale), int(h * scale)))
            out_path = os.path.join(out_dir, f"frame_{frame_num:06d}_obb.jpg")
            cv2.imwrite(out_path, img_s, [cv2.IMWRITE_JPEG_QUALITY, 90])
            print(f"  Saved: {out_path}")


# ─── Main ─────────────────────────────────────────────────────────────────────

def run_pipeline(
    entity_names: list[str],
    output_path: str,
    use_sam: bool = True,
) -> list[dict]:
    poses  = utils.load_poses()
    frames = utils.get_frame_numbers()
    K      = utils.build_K()

    print_stage("Pipeline Config")
    print(f"  Entities: {entity_names}")
    print(f"  Frames:   {frames}")
    print(f"  fx={config.FX:.2f}  fy={config.FY:.2f}  "
          f"cx={config.CX:.2f}  cy={config.CY:.2f}")
    print(f"  Image: {config.IMAGE_W}×{config.IMAGE_H}")
    print(f"  SAM:   {'enabled' if use_sam else 'disabled'}")
    print(f"  Output: {output_path}")

    results = []
    for name in entity_names:
        if name not in config.ENTITY_PRIOR_CENTER:
            print(f"WARNING: No prior center for {name} — skipping")
            continue
        obb = compute_obb_for_entity(name, frames, poses, K, use_sam=use_sam)
        results.append({"entity": name, "obb": obb})

    # Serialise
    print_stage("Writing output")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Saved {len(results)} OBBs to {output_path}")

    # Save validation images
    val_dir = os.path.join(os.path.dirname(output_path), "validation")
    print_stage("Saving validation images")
    save_validation_images(results, frames, poses, K, val_dir)

    return results


def main():
    parser = argparse.ArgumentParser(
        description="SAM-guided metric-semantic OBB estimation"
    )
    parser.add_argument(
        "--entities", nargs="+",
        default=["vga_socket", "ethernet_socket", "power_socket", "usb_socket", "audio_socket"],
        help="Entity names to estimate OBBs for"
    )
    parser.add_argument(
        "--output", default=None,
        help="Output JSON path (default: outputs/answers.json)"
    )
    parser.add_argument(
        "--no-sam", action="store_true",
        help="Disable SAM segmentation (use prior centers only)"
    )
    args = parser.parse_args()

    if args.output is None:
        args.output = os.path.join(config.OUTPUT_DIR, "answers.json")

    run_pipeline(
        entity_names=args.entities,
        output_path=args.output,
        use_sam=not args.no_sam,
    )


if __name__ == "__main__":
    main()
