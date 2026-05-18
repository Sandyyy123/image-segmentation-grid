"""
Grid detection and perspective correction for 10mm square grid backgrounds.

Workflow:
1. Detect grid lines via Hough transform or corner detection
2. Estimate homography from detected grid intersections
3. Validate orthogonality; reject images with severe perspective distortion
4. Compute pixel-per-mm scale factor from corrected image
"""

import cv2
import numpy as np
from typing import Optional


ORTHOGONALITY_THRESHOLD_DEG = 5.0  # reject if grid skew exceeds this


def detect_grid_corners(image: np.ndarray, grid_spacing_mm: float = 10.0) -> dict:
    """
    Detect the 10mm square grid in image and return:
      - homography matrix H (warps raw to rectified)
      - scale_px_per_mm (float)
      - is_valid (bool) — False if perspective is too severe
      - corrected_image (np.ndarray)
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image.copy()

    # Adaptive threshold to isolate dark grid lines on white paper
    thresh = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,
        blockSize=15, C=4
    )

    # Find grid line intersections using Harris corners
    corners_raw = cv2.goodFeaturesToTrack(
        thresh.astype(np.float32),
        maxCorners=500,
        qualityLevel=0.01,
        minDistance=15,
        blockSize=5
    )

    if corners_raw is None or len(corners_raw) < 9:
        return {"is_valid": False, "reason": "Insufficient grid corners detected"}

    corners = corners_raw.reshape(-1, 2)

    # Cluster corners onto a regular grid lattice
    H, scale, skew_deg = _fit_grid_homography(corners, grid_spacing_mm)

    if H is None:
        return {"is_valid": False, "reason": "Could not fit homography to grid corners"}

    if abs(skew_deg) > ORTHOGONALITY_THRESHOLD_DEG:
        return {
            "is_valid": False,
            "reason": f"Perspective skew {skew_deg:.1f}° exceeds {ORTHOGONALITY_THRESHOLD_DEG}° threshold",
            "skew_deg": skew_deg
        }

    h, w = image.shape[:2]
    corrected = cv2.warpPerspective(image, H, (w, h), flags=cv2.INTER_LINEAR)

    return {
        "is_valid": True,
        "homography": H,
        "scale_px_per_mm": scale,
        "skew_deg": skew_deg,
        "corrected_image": corrected,
        "corners_detected": len(corners)
    }


def _fit_grid_homography(corners: np.ndarray, grid_spacing_mm: float):
    """Fit a regular grid to detected corners; return (H, scale_px_per_mm, skew_deg)."""
    # Sort corners by x then y to build row/column structure
    xs = corners[:, 0]
    ys = corners[:, 1]

    # Estimate grid spacing in pixels by finding nearest-neighbor distances
    dists = []
    for pt in corners[:50]:  # sample first 50 points
        d = np.linalg.norm(corners - pt, axis=1)
        d = d[d > 5]
        if len(d):
            dists.append(np.min(d))

    if not dists:
        return None, None, None

    spacing_px = float(np.median(dists))
    scale_px_per_mm = spacing_px / grid_spacing_mm

    # Build ideal grid from bounding box of detected corners
    x_min, y_min = corners.min(axis=0)
    x_max, y_max = corners.max(axis=0)

    # Number of grid cells
    nx = max(2, int(round((x_max - x_min) / spacing_px)))
    ny = max(2, int(round((y_max - y_min) / spacing_px)))

    # Select 4 well-separated corners for homography
    src_pts = np.float32([
        [x_min, y_min],
        [x_max, y_min],
        [x_max, y_max],
        [x_min, y_max]
    ])

    dst_pts = np.float32([
        [0, 0],
        [nx * spacing_px, 0],
        [nx * spacing_px, ny * spacing_px],
        [0, ny * spacing_px]
    ])

    H, _ = cv2.findHomography(src_pts, dst_pts, method=0)

    # Estimate skew from homography column vectors
    col0 = H[:, 0]
    col1 = H[:, 1]
    cos_angle = np.dot(col0[:2], col1[:2]) / (
        np.linalg.norm(col0[:2]) * np.linalg.norm(col1[:2]) + 1e-9
    )
    skew_deg = float(np.degrees(np.arccos(np.clip(cos_angle, -1, 1))))
    skew_from_90 = abs(skew_deg - 90.0)

    return H, scale_px_per_mm, skew_from_90
