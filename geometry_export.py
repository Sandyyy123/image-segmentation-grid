"""
Export segmented item outlines to SVG and DXF formats.

SVG: uses svgwrite (pip install svgwrite)
DXF: uses ezdxf (pip install ezdxf)
Both outputs use real-world mm coordinates from the corrected image.
"""

import os
from typing import List, Optional, Tuple
from pathlib import Path

Contour = List[Tuple[float, float]]


# ---------------------------------------------------------------------------
# SVG export
# ---------------------------------------------------------------------------

def export_svg(
    contours_mm: List[Contour],
    output_path: str,
    page_width_mm: float = 210.0,
    page_height_mm: float = 297.0,
    stroke_color: str = "#0066cc",
    stroke_width_mm: float = 0.3
) -> str:
    """
    Write contours as closed SVG paths in mm units.
    Returns the path to the saved SVG file.
    """
    try:
        import svgwrite
    except ImportError:
        raise ImportError("pip install svgwrite")

    dwg = svgwrite.Drawing(
        output_path,
        size=(f"{page_width_mm}mm", f"{page_height_mm}mm"),
        viewBox=f"0 0 {page_width_mm} {page_height_mm}"
    )

    # Grid layer (faint 10mm squares for reference)
    grid = dwg.add(dwg.g(id="grid", stroke="#cccccc", stroke_width="0.1",
                          fill="none", opacity="0.5"))
    for x in range(0, int(page_width_mm), 10):
        grid.add(dwg.line((x, 0), (x, page_height_mm)))
    for y in range(0, int(page_height_mm), 10):
        grid.add(dwg.line((0, y), (page_width_mm, y)))

    # Item outlines layer
    items = dwg.add(dwg.g(id="items", stroke=stroke_color,
                           stroke_width=str(stroke_width_mm), fill="none"))

    for i, contour in enumerate(contours_mm):
        if len(contour) < 3:
            continue
        # SVG path: M x,y L x,y ... Z
        d = "M " + " L ".join(f"{x:.3f},{y:.3f}" for x, y in contour) + " Z"
        path = dwg.path(d=d, id=f"item_{i+1}")
        path["class"] = "item-outline"
        items.add(path)

    dwg.save()
    return output_path


# ---------------------------------------------------------------------------
# DXF export
# ---------------------------------------------------------------------------

def export_dxf(
    contours_mm: List[Contour],
    output_path: str,
    layer_name: str = "ITEM_OUTLINES"
) -> str:
    """
    Write contours as closed LWPOLYLINE entities in a DXF file.
    Units: millimeters. Returns path to saved DXF file.
    """
    try:
        import ezdxf
        from ezdxf.enums import TextEntityAlignment
    except ImportError:
        raise ImportError("pip install ezdxf")

    doc = ezdxf.new(dxfversion="R2010")
    doc.units = 4  # 4 = millimeters in DXF INSUNITS

    msp = doc.modelspace()

    # Add layer
    doc.layers.add(layer_name, color=5)  # 5 = blue in ACI

    for i, contour in enumerate(contours_mm):
        if len(contour) < 3:
            continue
        points_3d = [(x, y, 0.0) for x, y in contour]
        polyline = msp.add_lwpolyline(
            points_3d,
            dxfattribs={"layer": layer_name, "closed": True}
        )
        polyline.dxf.linetype = "CONTINUOUS"

    doc.saveas(output_path)
    return output_path


# ---------------------------------------------------------------------------
# Batch export
# ---------------------------------------------------------------------------

def export_all(
    contours_mm: List[Contour],
    output_dir: str,
    basename: str = "segmented_items",
    page_width_mm: float = 210.0,
    page_height_mm: float = 297.0
) -> dict:
    """
    Export both SVG and DXF for a list of item contours.
    Returns {"svg": path, "dxf": path}.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    svg_path = os.path.join(output_dir, f"{basename}.svg")
    dxf_path = os.path.join(output_dir, f"{basename}.dxf")

    results = {}

    try:
        export_svg(contours_mm, svg_path, page_width_mm, page_height_mm)
        results["svg"] = svg_path
        print(f"  SVG saved: {svg_path}")
    except Exception as e:
        print(f"  SVG export failed: {e}")
        results["svg"] = None

    try:
        export_dxf(contours_mm, dxf_path)
        results["dxf"] = dxf_path
        print(f"  DXF saved: {dxf_path}")
    except Exception as e:
        print(f"  DXF export failed: {e}")
        results["dxf"] = None

    return results
