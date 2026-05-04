"""
Camera geometry and data loading utilities.
"""

from __future__ import annotations

import json
import os

import cv2
import numpy as np

import config


# ── Data loading ───────────────────────────────────────────────────────────────

def load_poses() -> dict[int, np.ndarray]:
    """Return {frame_number: 4×4 camera-to-world matrix (float64)}."""
    path = os.path.join(config.DATA_DIR, "poses.json")
    with open(path) as f:
        raw = json.load(f)
    return {int(k): np.array(v, dtype=np.float64) for k, v in raw.items()}


def get_frame_numbers() -> list[int]:
    """Return sorted list of frame numbers with PNG images in DATA_DIR."""
    frames = []
    for fname in os.listdir(config.DATA_DIR):
        if fname.endswith(".png") and fname.startswith("frame_"):
            num = int(fname.split("_")[1].split(".")[0])
            frames.append(num)
    return sorted(frames)


def load_image_bgr(frame_number: int) -> np.ndarray:
    """Load frame as BGR numpy array (OpenCV format)."""
    path = os.path.join(config.DATA_DIR, f"frame_{frame_number:06d}.png")
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Image not found: {path}")
    return img


def load_image_rgb(frame_number: int) -> np.ndarray:
    """Load frame as RGB numpy array."""
    return load_image_bgr(frame_number)[:, :, ::-1].copy()


def build_K(fx=None, fy=None, cx=None, cy=None) -> np.ndarray:
    """Build 3×3 camera intrinsics matrix K."""
    return np.array([
        [fx or config.FX,            0,  cx or config.CX],
        [           0,  fy or config.FY,  cy or config.CY],
        [           0,             0,              1],
    ], dtype=np.float64)


# ── Camera geometry ────────────────────────────────────────────────────────────

def world_to_cam(p_world: np.ndarray, c2w: np.ndarray) -> np.ndarray:
    """
    Transform world point(s) to camera-space coordinates.
    c2w is a 4×4 camera-to-world matrix.
    Handles (3,) or (N,3) inputs.
    """
    R = c2w[:3, :3]
    t = c2w[:3, 3]
    if p_world.ndim == 1:
        return R.T @ (p_world - t)
    return (p_world - t) @ R  # equivalent: each row is R^T @ (p - t)


def project_to_image(
    p_world: np.ndarray,
    c2w: np.ndarray,
    K: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Project world point(s) to pixel coordinates.

    Args:
        p_world: (N,3) or (3,) world points
        c2w:     (4,4) camera-to-world matrix
        K:       (3,3) intrinsics

    Returns:
        pixels: (N,2) or (2,) [u, v] pixel coordinates
        depths: (N,) or scalar z in camera space
    """
    single = p_world.ndim == 1
    pts = np.atleast_2d(p_world).astype(np.float64)
    cam = world_to_cam(pts, c2w)           # (N,3)
    depths = cam[:, 2]
    with np.errstate(divide='ignore', invalid='ignore'):
        u = K[0, 0] * cam[:, 0] / cam[:, 2] + K[0, 2]
        v = K[1, 1] * cam[:, 1] / cam[:, 2] + K[1, 2]
    pixels = np.stack([u, v], axis=-1)    # (N,2)
    if single:
        return pixels[0], float(depths[0])
    return pixels, depths


def is_visible(
    p_world: np.ndarray,
    c2w: np.ndarray,
    K: np.ndarray,
    margin: int = 50,
) -> bool:
    """Return True if the world point projects within the image bounds."""
    pix, depth = project_to_image(p_world, c2w, K)
    if depth <= 0:
        return False
    u, v = float(pix[0]), float(pix[1])
    return (margin <= u < config.IMAGE_W - margin and
            margin <= v < config.IMAGE_H - margin)


def backproject(
    pixels_uv: np.ndarray,
    depths: np.ndarray,
    K: np.ndarray,
    c2w: np.ndarray,
) -> np.ndarray:
    """
    Back-project (N,2) pixel coords + (N,) metric depths → (N,3) world points.
    """
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    u, v = pixels_uv[:, 0], pixels_uv[:, 1]
    x_cam = (u - cx) / fx * depths
    y_cam = (v - cy) / fy * depths
    pts_cam = np.stack([x_cam, y_cam, depths], axis=-1)  # (N,3) in camera space
    R = c2w[:3, :3]
    t = c2w[:3, 3]
    return pts_cam @ R.T + t   # transform to world space


# ── OBB utilities ─────────────────────────────────────────────────────────────

def obb_corners_3d(center, extent, rotation) -> np.ndarray:
    """
    Compute the 8 corners of an OBB.
    center:   (3,) world-space centre
    extent:   (3,) FULL edge lengths along each OBB axis (metres)
    rotation: (3,3) columns = OBB axes in world space

    Corners = center + R * (± extent/2)
    Returns (8,3) corner coordinates.
    """
    c = np.array(center)
    e = np.array(extent) / 2.0   # full length → half-extent
    R = np.array(rotation)
    corners = []
    for sx in (-1, 1):
        for sy in (-1, 1):
            for sz in (-1, 1):
                corners.append(c + R @ (e * np.array([sx, sy, sz])))
    return np.array(corners)   # (8,3)


def project_obb_to_image(
    obb: dict,
    c2w: np.ndarray,
    K: np.ndarray,
) -> np.ndarray | None:
    """
    Project all 8 OBB corners to image plane.
    Returns (8,2) pixel array, or None if all corners are behind the camera.
    """
    corners = obb_corners_3d(obb["center"], obb["extent"], obb["rotation"])
    pixels, depths = project_to_image(corners, c2w, K)
    if (depths <= 0).all():
        return None
    return pixels


def draw_obb_on_image(
    img_bgr: np.ndarray,
    obb: dict,
    c2w: np.ndarray,
    K: np.ndarray,
    label: str = "",
    color: tuple = (0, 255, 0),
    thickness: int = 3,
) -> np.ndarray:
    """Draw projected OBB wireframe on a BGR image (returns a copy)."""
    vis = img_bgr.copy()
    pixels = project_obb_to_image(obb, c2w, K)
    if pixels is None:
        return vis

    corners_2d = pixels.astype(int)
    H, W = vis.shape[:2]

    def clip_pt(p):
        return (int(np.clip(p[0], 0, W - 1)), int(np.clip(p[1], 0, H - 1)))

    # Draw edges: two corners share an edge iff their 3-bit indices differ in 1 bit
    for i in range(8):
        for j in range(i + 1, 8):
            if bin(i ^ j).count("1") == 1:
                cv2.line(vis, clip_pt(corners_2d[i]), clip_pt(corners_2d[j]),
                         color, thickness)

    if label:
        cx = int(np.clip(corners_2d[:, 0].mean(), 0, W - 1))
        cy = int(np.clip(corners_2d[:, 1].mean(), 0, H - 1))
        cv2.putText(vis, label, (cx, cy - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 2)
    return vis
