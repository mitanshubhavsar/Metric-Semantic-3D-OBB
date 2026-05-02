"""
Multi-view triangulation via Direct Linear Transform (DLT/SVD).

Given 2D observations of a point across N frames with known camera
intrinsics and poses, recover the 3D world position.
"""

from __future__ import annotations

import numpy as np


def triangulate_dlt(
    pixels: list[tuple[float, float]],
    c2ws: list[np.ndarray],
    K: np.ndarray,
) -> np.ndarray | None:
    """
    Triangulate a 3D point from N ≥ 2 observations using DLT (SVD).

    Args:
        pixels: list of (u, v) pixel observations
        c2ws:   list of (4,4) camera-to-world matrices
        K:      (3,3) camera intrinsics

    Returns:
        (3,) world-space point, or None if degenerate.
    """
    if len(pixels) < 2:
        return None

    rows = []
    for (u, v), c2w in zip(pixels, c2ws):
        R = c2w[:3, :3]
        t = c2w[:3, 3]
        # World-to-camera projection matrix P = K @ [R^T | -R^T t]
        Rt = np.zeros((3, 4))
        Rt[:3, :3] = R.T
        Rt[:3, 3]  = -R.T @ t
        P = K @ Rt  # (3,4)
        # DLT: u*(P2 X) = P0 X  →  u*P2 - P0 = 0 (one row per coord)
        rows.append(u * P[2] - P[0])
        rows.append(v * P[2] - P[1])

    A = np.array(rows)        # (2N, 4)
    _, _, Vt = np.linalg.svd(A)
    X = Vt[-1]                # homogeneous solution
    if abs(X[3]) < 1e-10:
        return None
    return X[:3] / X[3]


def reprojection_error(
    pt_world: np.ndarray,
    pixels: list[tuple[float, float]],
    c2ws: list[np.ndarray],
    K: np.ndarray,
) -> float:
    """Mean reprojection error (pixels) of a 3D point across observations."""
    from utils import project_to_image
    errs = []
    for (u, v), c2w in zip(pixels, c2ws):
        pix, depth = project_to_image(pt_world, c2w, K)
        if depth <= 0:
            continue
        errs.append(float(np.sqrt((pix[0] - u) ** 2 + (pix[1] - v) ** 2)))
    return float(np.mean(errs)) if errs else 1e9


def triangulate_box_corners(
    boxes: list[list[float]],
    c2ws: list[np.ndarray],
    K: np.ndarray,
) -> np.ndarray:
    """
    Triangulate a cloud of 3D points from 2D bounding boxes.

    Strategy:
    - Triangulate box centre from all pairs of frames
    - Also triangulate the 4 box corners from all pairs
    This gives a dense cloud that OBB fitting can use.

    Returns (N, 3) array of 3D points.
    """
    pts = []
    n = len(boxes)
    if n < 2:
        return np.zeros((0, 3))

    # Sampling points within each box: centre + 4 corners + 4 edge midpoints
    def box_samples(box):
        x1, y1, x2, y2 = box
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        return [
            (cx, cy),
            (x1, y1), (x2, y1), (x1, y2), (x2, y2),
            (cx, y1), (cx, y2), (x1, cy), (x2, cy),
        ]

    for i in range(n):
        for j in range(i + 1, n):
            samples_i = box_samples(boxes[i])
            samples_j = box_samples(boxes[j])
            for si, sj in zip(samples_i, samples_j):
                pt = triangulate_dlt([si, sj], [c2ws[i], c2ws[j]], K)
                if pt is not None:
                    pts.append(pt)

    return np.array(pts) if pts else np.zeros((0, 3))
