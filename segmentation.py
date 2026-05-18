"""
Item segmentation on white-paper backgrounds.

Strategy:
  1. Primary: Meta SAM (Segment Anything Model) via transformers/segment-anything
  2. Fallback: OpenCV GrabCut + contour extraction (no GPU required)

Both paths return binary masks and contour polygons (list of (x,y) tuples).
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

Contour = List[Tuple[float, float]]


# ---------------------------------------------------------------------------
# SAM-based segmentation
# ---------------------------------------------------------------------------

def segment_with_sam(image_rgb: np.ndarray, device: str = "cpu") -> List[np.ndarray]:
    """
    Run Meta SAM automatic mask generation. Returns list of binary masks
    (H x W bool arrays), one per detected item.

    Requires: pip install segment-anything torch torchvision
    Model checkpoint: sam_vit_h_4b8939.pth (download separately)
    """
    try:
        from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
        import torch

        checkpoint = "sam_vit_h_4b8939.pth"
        sam = sam_model_registry["vit_h"](checkpoint=checkpoint)
        sam.to(device=device)

        generator = SamAutomaticMaskGenerator(
            model=sam,
            points_per_side=32,
            pred_iou_thresh=0.88,
            stability_score_thresh=0.95,
            min_mask_region_area=500,
        )
        raw_masks = generator.generate(image_rgb)

        # Filter out the full-image background mask
        h, w = image_rgb.shape[:2]
        total_px = h * w
        masks = [
            m["segmentation"]
            for m in raw_masks
            if m["area"] < total_px * 0.85 and m["area"] > 200
        ]
        logger.info(f"SAM: {len(masks)} item masks found")
        return masks

    except ImportError:
        logger.warning("segment-anything not installed; falling back to GrabCut")
        return []
    except FileNotFoundError:
        logger.warning("SAM checkpoint not found; falling back to GrabCut")
        return []


# ---------------------------------------------------------------------------
# GrabCut fallback (no deep learning dependency)
# ---------------------------------------------------------------------------

def segment_with_grabcut(image_bgr: np.ndarray) -> List[np.ndarray]:
    """
    Segment items from white paper using:
      1. White background subtraction
      2. Morphological cleanup
      3. Connected component isolation
      4. GrabCut refinement per component

    Returns list of binary masks (H x W uint8, values 0 or 255).
    """
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    # Threshold: white paper is ~240+, items are darker
    _, fg_mask = cv2.threshold(gray, 220, 255, cv2.THRESH_BINARY_INV)

    # Remove grid lines (thin dark lines ~1-3px wide)
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    cleaned = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel_open, iterations=2)

    # Close small gaps within items
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    closed = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel_close, iterations=3)

    # Connected components
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(closed, connectivity=8)

    masks = []
    h, w = image_bgr.shape[:2]

    for label in range(1, num_labels):
        area = stats[label, cv2.CC_STAT_AREA]
        if area < 500:  # skip noise
            continue

        component_mask = (labels == label).astype(np.uint8) * 255

        # GrabCut refinement using the component bounding box
        x = stats[label, cv2.CC_STAT_LEFT]
        y = stats[label, cv2.CC_STAT_TOP]
        bw = stats[label, cv2.CC_STAT_WIDTH]
        bh = stats[label, cv2.CC_STAT_HEIGHT]

        rect = (max(0, x - 5), max(0, y - 5),
                min(bw + 10, w - x), min(bh + 10, h - y))

        bgd_model = np.zeros((1, 65), np.float64)
        fgd_model = np.zeros((1, 65), np.float64)
        gc_mask = np.where(component_mask > 0,
                           cv2.GC_PR_FGD, cv2.GC_BGD).astype(np.uint8)

        try:
            cv2.grabCut(image_bgr, gc_mask, rect, bgd_model, fgd_model,
                        iterCount=5, mode=cv2.GC_INIT_WITH_MASK)
            refined = np.where(
                (gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD),
                255, 0
            ).astype(np.uint8)
        except cv2.error:
            refined = component_mask

        masks.append(refined)

    logger.info(f"GrabCut: {len(masks)} item masks found")
    return masks


# ---------------------------------------------------------------------------
# Contour extraction
# ---------------------------------------------------------------------------

def mask_to_contour(mask: np.ndarray, simplify_epsilon: float = 1.5) -> Optional[Contour]:
    """
    Convert binary mask to a simplified polygon contour.
    Returns list of (x, y) float tuples, or None if no contour found.
    """
    m = mask.astype(np.uint8)
    if m.max() == 0:
        return None

    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None

    # Take the largest contour
    largest = max(contours, key=cv2.contourArea)

    # Douglas-Peucker simplification
    eps = simplify_epsilon * cv2.arcLength(largest, True) / len(largest)
    approx = cv2.approxPolyDP(largest, eps, True)

    return [(float(p[0][0]), float(p[0][1])) for p in approx]


# ---------------------------------------------------------------------------
# Main entry
# ---------------------------------------------------------------------------

def run_segmentation(
    image_bgr: np.ndarray,
    use_sam: bool = True,
    device: str = "cpu",
    scale_px_per_mm: Optional[float] = None
) -> List[dict]:
    """
    Full segmentation pipeline. Returns list of dicts:
      {
        "mask": np.ndarray,
        "contour_px": Contour,            # polygon in pixel coords
        "contour_mm": Contour or None,    # polygon in mm coords (if scale known)
        "area_mm2": float or None,
        "bbox_px": (x, y, w, h)
      }
    """
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    masks = []
    if use_sam:
        masks = segment_with_sam(image_rgb, device=device)

    if not masks:
        masks_uint8 = segment_with_grabcut(image_bgr)
        # Convert bool SAM masks to bool for uniform handling
        masks = [m.astype(bool) for m in masks_uint8]

    results = []
    for mask in masks:
        mask_u8 = (mask.astype(np.uint8) * 255) if mask.dtype == bool else mask
        contour_px = mask_to_contour(mask_u8)
        if contour_px is None:
            continue

        contour_mm = None
        area_mm2 = None
        if scale_px_per_mm and scale_px_per_mm > 0:
            s = scale_px_per_mm
            contour_mm = [(x / s, y / s) for x, y in contour_px]
            area_mm2 = float(cv2.contourArea(
                np.array(contour_px, dtype=np.float32).reshape(-1, 1, 2)
            ) / (s * s))

        cnts, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        bbox_px = cv2.boundingRect(cnts[0]) if cnts else (0, 0, 0, 0)

        results.append({
            "mask": mask_u8,
            "contour_px": contour_px,
            "contour_mm": contour_mm,
            "area_mm2": area_mm2,
            "bbox_px": bbox_px
        })

    return results
