"""
OBB (Oriented Bounding Box) fitting from 3D point clouds.

Two methods:
  1. Open3D's minimal OBB (most robust, preferred).
  2. PCA-based OBB (fallback).

Both return the same dict format:
    center:   [cx, cy, cz]         world-space centre
    extent:   [ex, ey, ez]         half-extents along each axis
    rotation: [[r00..],[r10..],[r20..]]  columns = OBB local axes in world space
"""

from __future__ import annotations

import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors

import config


def remove_outliers(
    pts: np.ndarray,
    nb_neighbors: int = config.OUTLIER_NB_NEIGHBORS,
    std_ratio: float = config.OUTLIER_STD_RATIO,
) -> np.ndarray:
    """Statistical outlier removal (k-NN mean distance filter)."""
    if len(pts) <= nb_neighbors + 1:
        return pts
    nbrs = NearestNeighbors(n_neighbors=nb_neighbors + 1).fit(pts)
    dists, _ = nbrs.kneighbors(pts)
    mean_d = dists[:, 1:].mean(axis=1)
    thresh = mean_d.mean() + std_ratio * mean_d.std()
    return pts[mean_d <= thresh]


def keep_largest_cluster(
    pts: np.ndarray,
    eps: float = config.DBSCAN_EPS_M,
    min_samples: int = config.DBSCAN_MIN_SAMPLES,
) -> np.ndarray:
    """DBSCAN clustering – keep the largest non-noise cluster."""
    if len(pts) < min_samples:
        return pts
    labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(pts)
    valid = labels[labels >= 0]
    if len(valid) == 0:
        return pts
    unique, counts = np.unique(valid, return_counts=True)
    best = unique[counts.argmax()]
    return pts[labels == best]


def fit_pca_obb(pts: np.ndarray) -> dict:
    """Fit OBB via PCA of the point cloud."""
    center = pts.mean(axis=0)
    pts_c = pts - center
    cov = pts_c.T @ pts_c / len(pts_c)
    eigvals, eigvecs = np.linalg.eigh(cov)
    # Sort descending by variance
    idx = np.argsort(eigvals)[::-1]
    eigvecs = eigvecs[:, idx]

    proj = pts_c @ eigvecs
    half_extents = (proj.max(axis=0) - proj.min(axis=0)) / 2.0
    # Refine centre to midpoint of bbox in PCA space
    center = center + eigvecs @ ((proj.max(axis=0) + proj.min(axis=0)) / 2.0)

    return {
        "center":   center.tolist(),
        "extent":   half_extents.tolist(),
        "rotation": eigvecs.tolist(),
    }


def fit_open3d_obb(pts: np.ndarray) -> dict | None:
    """Fit minimal OBB using Open3D (more robust than PCA)."""
    try:
        import open3d as o3d
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts)
        obb = pcd.get_minimal_oriented_bounding_box(robust=True)
        R = np.array(obb.R)            # (3,3) rotation, columns = axes
        extent = np.array(obb.extent) / 2.0  # half-extents
        center = np.array(obb.center)
        return {
            "center":   center.tolist(),
            "extent":   extent.tolist(),
            "rotation": R.tolist(),
        }
    except Exception as e:
        print(f"  [obb] Open3D failed: {e}")
        return None


def fit_obb(pts: np.ndarray, use_open3d: bool = True) -> dict | None:
    """
    Full OBB pipeline: outlier removal → clustering → OBB fit.

    Returns OBB dict or None if not enough points.
    """
    if len(pts) < config.MIN_POINTS_FOR_OBB:
        print(f"  [obb] {len(pts)} pts < min {config.MIN_POINTS_FOR_OBB}")
        return None

    pts = remove_outliers(pts)
    print(f"  [obb] After outlier removal: {len(pts)} pts")

    pts = keep_largest_cluster(pts)
    print(f"  [obb] After clustering: {len(pts)} pts")

    if len(pts) < 4:
        return None

    if use_open3d:
        result = fit_open3d_obb(pts)
        if result is not None:
            return result

    return fit_pca_obb(pts)


def make_obb_from_center_extent_rotation(
    center: list | np.ndarray,
    extent: list | np.ndarray,
    rotation: list | np.ndarray,
) -> dict:
    """Pack center, extent (half), rotation into OBB dict."""
    return {
        "center":   list(np.array(center).tolist()),
        "extent":   list(np.array(extent).tolist()),
        "rotation": np.array(rotation).tolist(),
    }
