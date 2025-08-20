#!/usr/bin/env python3
"""
rebuild_pngs_from_tif.py
────────────────────────
Tasks performed:

1. **TIFF → PNG conversion**
   Recursively scan every “*.TIF / *.tif” under
   uxb_tpi_clr_with_masks/**/images/ and create an 8‑bit PNG with the
   same stem in the same folder (re‑placing any existing copy).

2. **Orphan‑mask cleanup**
   For every mask PNG under uxb_tpi_clr_with_masks/**/masks/, keep it
   only when the trailing numeric ID in its filename matches the
   trailing numeric ID of at least one image PNG.  All others are
   deleted.

Run:

    python rebuild_pngs_from_tif.py
"""

from pathlib import Path
from typing import Optional
import re
import numpy as np
from PIL import Image

# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------
ROOT = Path("uxb_tpi_clr_with_masks").resolve()   # dataset root
IMG_SUB = "images"                               # sub‑folder containing images
MSK_SUB = "masks"                                # sub‑folder containing masks
ID_RE   = re.compile(r"(\d+)$")                  # regex to capture trailing digits

# ---------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------
def scale_to_8bit(arr: np.ndarray) -> np.ndarray:
    """
    Rescale 16‑bit or float pixel values to 0‑255 uint8 per band.
    If the array is already uint8, it is returned unmodified.
    """
    if arr.dtype == np.uint8:
        return arr
    if arr.ndim == 2:                 # single band → add dummy channel dim
        arr = arr[..., None]

    out = np.empty_like(arr, dtype=np.uint8)
    for c in range(arr.shape[2]):
        band = arr[..., c].astype(np.float32)
        lo, hi = np.percentile(band, (2, 98))
        if hi <= lo:                  # nearly constant band → all black
            out[..., c] = 0
            continue
        band = (band - lo) / (hi - lo) * 255.0
        out[..., c] = np.clip(band, 0, 255)

    return out.squeeze() if out.shape[2] == 1 else out


def convert_tif(tif_path: Path) -> None:
    """
    Convert one .tif/.TIF file to an 8‑bit PNG next to it.
    Existing PNG with the same stem is overwritten.
    """
    png_path = tif_path.with_suffix(".png")
    if png_path.exists():
        png_path.unlink()

    with Image.open(tif_path) as im:
        arr8  = scale_to_8bit(np.array(im))
        mode  = "L" if arr8.ndim == 2 else "RGB"
        Image.fromarray(arr8, mode).save(png_path, "PNG", compress_level=3)

    print(f"✓ {png_path.relative_to(ROOT)}")


def trailing_id(path: Path) -> Optional[str]:
    """
    Extract trailing digits from a file’s stem (e.g. “xxx123” → “123”).
    Returns None if no trailing digits are found.
    """
    m = ID_RE.search(path.stem)
    return m.group(1) if m else None

# ---------------------------------------------------------------------
# Main routine
# ---------------------------------------------------------------------
def main() -> None:
    if not ROOT.is_dir():
        raise SystemExit(f"{ROOT} does not exist.")

    # ----- Part 1 – TIFF → PNG ------------------------------------------------
    tifs = [p for p in ROOT.rglob("*.tif") if IMG_SUB in p.parts] + \
           [p for p in ROOT.rglob("*.TIF") if IMG_SUB in p.parts]

    if tifs:
        print(f"Converting {len(tifs)} TIFFs → PNG …")
        for tif in sorted(tifs):
            convert_tif(tif)
    else:
        print("No .tif files found; skipping TIFF → PNG conversion.")

    # ----- Part 2 – Remove orphan masks ---------------------------------------
    # 2a. Collect numeric IDs of all existing image PNGs
    img_pngs = [p for p in ROOT.rglob("*.png") if IMG_SUB in p.parts]
    img_ids  = {tid for p in img_pngs if (tid := trailing_id(p))}

    # 2b. Remove masks whose numeric ID is not in the image‑ID set
    mask_pngs = [p for p in ROOT.rglob("*.png") if MSK_SUB in p.parts]
    removed = 0
    for msk in mask_pngs:
        m_id = trailing_id(msk)
        if m_id is None or m_id not in img_ids:
            msk.unlink()
            removed += 1
            print(f"✗ removed extra mask {msk.relative_to(ROOT)}")

    print(f"Done.  Orphan masks removed: {removed}")

# ---------------------------------------------------------------------
if __name__ == "__main__":
    main()
