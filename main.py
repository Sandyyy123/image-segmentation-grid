"""
Image Segmentation Pipeline — 10mm Grid Items
==============================================
Process phone-captured images of items on white grid paper:
  1. Detect 10mm grid -> compute scale (px/mm) and correct perspective
  2. Segment items using SAM (primary) or GrabCut (fallback)
  3. Export outlines as SVG + DXF in mm coordinates

Usage:
    python main.py --input images/ --output results/
    python main.py --input photo.jpg --output results/ --no-sam
    python main.py --input images/ --output results/ --device cuda
"""

import argparse
import os
import sys
import json
import cv2
from pathlib import Path

from grid_detection import detect_grid_corners
from segmentation import run_segmentation
from geometry_export import export_all

SUPPORTED_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}


def process_image(
    image_path: str,
    output_dir: str,
    use_sam: bool = True,
    device: str = "cpu",
    verbose: bool = True
) -> dict:
    """Process a single image through the full pipeline."""
    name = Path(image_path).stem
    out_subdir = os.path.join(output_dir, name)
    Path(out_subdir).mkdir(parents=True, exist_ok=True)

    if verbose:
        print(f"\n{'='*60}")
        print(f"Processing: {image_path}")

    image = cv2.imread(image_path)
    if image is None:
        return {"image": image_path, "status": "error", "reason": "Cannot read image"}

    # Step 1: Grid detection + perspective correction
    grid_result = detect_grid_corners(image)

    if not grid_result["is_valid"]:
        reason = grid_result.get("reason", "Unknown grid detection failure")
        if verbose:
            print(f"  REJECTED: {reason}")
        return {
            "image": image_path,
            "status": "rejected",
            "reason": reason
        }

    scale = grid_result["scale_px_per_mm"]
    corrected = grid_result["corrected_image"]
    skew = grid_result.get("skew_deg", 0.0)

    if verbose:
        print(f"  Grid: {grid_result['corners_detected']} corners | "
              f"Scale: {scale:.2f} px/mm | Skew: {skew:.2f}°")

    # Save corrected image for inspection
    cv2.imwrite(os.path.join(out_subdir, f"{name}_corrected.jpg"), corrected)

    # Step 2: Segmentation
    items = run_segmentation(corrected, use_sam=use_sam, device=device,
                              scale_px_per_mm=scale)

    if verbose:
        print(f"  Segments: {len(items)} items found")

    if not items:
        return {
            "image": image_path,
            "status": "no_items",
            "scale_px_per_mm": scale,
            "skew_deg": skew
        }

    # Overlay contours on corrected image for QA
    overlay = corrected.copy()
    for i, item in enumerate(items):
        import numpy as np
        pts = np.array(item["contour_px"], dtype=np.int32).reshape(-1, 1, 2)
        cv2.drawContours(overlay, [pts], -1, (0, 200, 100), 2)
        x, y, w, h = item["bbox_px"]
        cv2.putText(overlay, f"#{i+1}", (x, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 100, 255), 2)
    cv2.imwrite(os.path.join(out_subdir, f"{name}_overlay.jpg"), overlay)

    # Step 3: Export geometry
    contours_mm = [item["contour_mm"] for item in items if item["contour_mm"]]
    export_result = export_all(contours_mm, out_subdir, basename=name)

    # Save metadata JSON
    meta = {
        "image": image_path,
        "status": "ok",
        "scale_px_per_mm": round(scale, 4),
        "skew_deg": round(skew, 3),
        "items_found": len(items),
        "items": [
            {
                "id": i + 1,
                "area_mm2": round(item["area_mm2"], 2) if item["area_mm2"] else None,
                "bbox_px": list(item["bbox_px"]),
                "contour_points": len(item["contour_px"])
            }
            for i, item in enumerate(items)
        ],
        "outputs": export_result
    }

    with open(os.path.join(out_subdir, f"{name}_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    if verbose:
        areas = [i["area_mm2"] for i in meta["items"] if i["area_mm2"]]
        if areas:
            print(f"  Areas: {[f'{a:.1f}mm²' for a in areas]}")
        if export_result.get("svg"):
            print(f"  SVG: {export_result['svg']}")
        if export_result.get("dxf"):
            print(f"  DXF: {export_result['dxf']}")

    return meta


def main():
    parser = argparse.ArgumentParser(description="Segment items on 10mm grid paper")
    parser.add_argument("--input", required=True, help="Image file or directory")
    parser.add_argument("--output", default="results", help="Output directory")
    parser.add_argument("--no-sam", action="store_true", help="Skip SAM, use GrabCut only")
    parser.add_argument("--device", default="cpu", help="SAM device: cpu or cuda")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    # Collect input images
    input_path = Path(args.input)
    if input_path.is_file():
        images = [str(input_path)]
    elif input_path.is_dir():
        images = [
            str(p) for p in sorted(input_path.iterdir())
            if p.suffix.lower() in SUPPORTED_EXT
        ]
    else:
        print(f"Error: {args.input} not found")
        sys.exit(1)

    if not images:
        print("No supported image files found")
        sys.exit(1)

    print(f"Processing {len(images)} image(s) -> {args.output}/")

    results = []
    for img_path in images:
        result = process_image(
            img_path,
            args.output,
            use_sam=not args.no_sam,
            device=args.device,
            verbose=not args.quiet
        )
        results.append(result)

    # Summary
    ok = sum(1 for r in results if r.get("status") == "ok")
    rejected = sum(1 for r in results if r.get("status") == "rejected")
    no_items = sum(1 for r in results if r.get("status") == "no_items")

    print(f"\n{'='*60}")
    print(f"Summary: {ok} processed | {rejected} rejected (perspective) | "
          f"{no_items} no items found | {len(images)} total")

    # Save full run summary
    summary_path = os.path.join(args.output, "run_summary.json")
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Summary saved: {summary_path}")


if __name__ == "__main__":
    main()
