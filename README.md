# Image Segmentation — 10mm Grid Items

Python pipeline for segmenting items photographed on white paper with a 10mm square grid. Handles uncontrolled phone camera inputs: variable lighting, optics, and perspective.

## Pipeline

```
Phone image
    │
    ▼
┌─────────────────────────┐
│  Grid Detection          │  Detect 10mm grid corners
│  + Perspective Correction│  Compute homography, validate orthogonality
│  + Scale Estimation      │  → px/mm scale factor
└──────────┬──────────────┘
           │ reject if skew > 5°
           ▼
┌─────────────────────────┐
│  Segmentation            │  SAM (primary) or GrabCut (fallback)
│  Item Isolation          │  Binary mask per item
│  Contour Extraction      │  Douglas-Peucker simplification
└──────────┬──────────────┘
           ▼
┌─────────────────────────┐
│  Geometry Export         │  SVG (svgwrite) + DXF (ezdxf)
│  mm-coordinate outlines  │  Real-world dimensions
│  QA overlay image        │  Annotated JPEG for inspection
└─────────────────────────┘
```

## Setup

```bash
pip install -r requirements.txt

# Optional: SAM (GPU recommended)
pip install torch torchvision
pip install git+https://github.com/facebookresearch/segment-anything.git
# Download checkpoint:
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
```

## Usage

```bash
# Process a folder of images (SAM if available, GrabCut fallback)
python main.py --input images/ --output results/

# Single image, GrabCut only (no GPU required)
python main.py --input photo.jpg --output results/ --no-sam

# SAM on GPU
python main.py --input images/ --output results/ --device cuda
```

## Output per image

```
results/
└── photo_name/
    ├── photo_name_corrected.jpg   # perspective-corrected image
    ├── photo_name_overlay.jpg     # segmentation QA overlay
    ├── photo_name.svg             # item outlines (mm, with grid reference)
    ├── photo_name.dxf             # item outlines (mm, LWPOLYLINE entities)
    └── photo_name_meta.json       # scale, skew, per-item area + bbox
```

## Rejection criteria

Images are rejected (not processed) when:
- Fewer than 9 grid corners detected (too blurry / no grid visible)
- Grid skew exceeds 5° from orthogonal (perspective too severe)

Rejected images are logged in `run_summary.json` with the reason.

## Key modules

| Module | Responsibility |
|--------|---------------|
| `grid_detection.py` | Hough/Harris corner detection, homography fit, scale estimation |
| `segmentation.py` | SAM mask generation + GrabCut fallback, contour extraction |
| `geometry_export.py` | SVG (svgwrite) and DXF (ezdxf) export in mm coordinates |
| `main.py` | CLI entry point, batch processing, JSON summary |

## Dependencies

- **OpenCV** — grid detection, GrabCut, morphological ops
- **svgwrite** — SVG export
- **ezdxf** — DXF export
- **SAM** (optional) — Meta's Segment Anything Model for higher accuracy
