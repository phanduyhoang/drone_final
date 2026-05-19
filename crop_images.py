"""
Crop large aerial images into experiment-ready regions.
Run once before experiments from the thesis_experiments/ directory.

Outputs (all in images/):
  corner_crop_terrain.tif   -- 9000x8000 terrain crop from Corner_orthoimage.tif
  summer_city_crop.tif      -- 9000x8000 city crop from summer_city.tif
  snow_crop.tif             -- clean crop of snow/elevation image, uint8 BGR

Usage:
    cd thesis_experiments
    python crop_images.py
"""
import cv2
import os
import numpy as np

out_dir = "images"
os.makedirs(out_dir, exist_ok=True)


def load_as_bgr_uint8(path):
    """
    Load any image as uint8 BGR.
    Handles standard 8-bit images AND 16/32/64-bit float GeoTIFFs.
    Returns (img_bgr, was_normalized) or (None, False) on failure.
    """
    # Try standard color load first (works for most images incl. uint8 TIFFs)
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is not None:
        return img, False

    # Fallback: load raw (handles float32/float64 GeoTIFFs)
    img_raw = cv2.imread(path, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_ANYCOLOR)
    if img_raw is None:
        img_raw = cv2.imread(path, cv2.IMREAD_ANYDEPTH)
    if img_raw is None:
        return None, False

    # If 3D, collapse to 2D by taking mean across channels
    if img_raw.ndim == 3:
        img_f = img_raw.mean(axis=2)
    else:
        img_f = img_raw.copy()

    img_f = img_f.astype(np.float64)

    # Replace NaN/inf with 0
    bad = ~np.isfinite(img_f)
    img_f[bad] = np.nan

    finite = img_f[np.isfinite(img_f)]
    if finite.size == 0:
        print("  WARNING: no finite pixels found — image may be all NaN")
        img_u8 = np.zeros(img_raw.shape[:2], dtype=np.uint8)
    else:
        vmin, vmax = finite.min(), finite.max()
        span = vmax - vmin
        if span < 1e-9:
            img_u8 = np.zeros(img_raw.shape[:2], dtype=np.uint8)
        else:
            norm = (img_f - vmin) / span * 255.0
            norm[~np.isfinite(norm)] = 0.0
            img_u8 = np.clip(norm, 0, 255).astype(np.uint8)

    # Convert grayscale to BGR so downstream IMREAD_COLOR gets the same format
    img_bgr = cv2.cvtColor(img_u8, cv2.COLOR_GRAY2BGR)
    return img_bgr, True


# ==============================================================
# 1. Terrain crop from Corner_orthoimage.tif
# ==============================================================
src_terrain = os.path.join(out_dir, "Corner_orthoimage.tif")
if not os.path.exists(src_terrain):
    print(f"WARNING: {src_terrain} not found — skipping terrain crop.")
else:
    print(f"Loading terrain image: {src_terrain} (may take a moment)...")
    img, normalized = load_as_bgr_uint8(src_terrain)
    if img is None:
        print(f"  ERROR: Could not load {src_terrain}")
    else:
        H, W = img.shape[:2]
        print(f"  Loaded: {W}x{H}{' [normalized to uint8]' if normalized else ''}")
        CW, CH = 9000, 8000
        cx, cy = 15000, 11000  # shifted right/down to avoid black padding
        x1 = max(0, cx - CW // 2)
        y1 = max(0, cy - CH // 2)
        x2 = min(W, x1 + CW)
        y2 = min(H, y1 + CH)
        x1 = max(0, x2 - CW)
        y1 = max(0, y2 - CH)
        crop = img[y1:y2, x1:x2]
        out_path = os.path.join(out_dir, "corner_crop_terrain.tif")
        cv2.imwrite(out_path, crop)
        print(f"  Saved {out_path}  size={crop.shape[1]}x{crop.shape[0]}  region=[{x1}:{x2}, {y1}:{y2}]")

# ==============================================================
# 2. Summer city crop (center of image)
# ==============================================================
src_city = os.path.join(out_dir, "summer_city.tif")
if not os.path.exists(src_city):
    print(f"WARNING: {src_city} not found — skipping city crop.")
else:
    print(f"\nLoading city image: {src_city}...")
    img_city, normalized = load_as_bgr_uint8(src_city)
    if img_city is None:
        print(f"  ERROR: Could not load {src_city}")
    else:
        H2, W2 = img_city.shape[:2]
        print(f"  Loaded: {W2}x{H2}{' [normalized to uint8]' if normalized else ''}")
        CW, CH = 9000, 8000
        cx2, cy2 = W2 // 2, H2 // 2   # dead center
        x1 = max(0, cx2 - CW // 2);  x2 = min(W2, x1 + CW);  x1 = max(0, x2 - CW)
        y1 = max(0, cy2 - CH // 2);  y2 = min(H2, y1 + CH);  y1 = max(0, y2 - CH)
        crop_city = img_city[y1:y2, x1:x2]
        out_city = os.path.join(out_dir, "summer_city_crop.tif")
        cv2.imwrite(out_city, crop_city)
        print(f"  Saved {out_city}  size={crop_city.shape[1]}x{crop_city.shape[0]}  region=[{x1}:{x2}, {y1}:{y2}]")

# ==============================================================
# 3. Snow crop
#    The snow image is a float64 GeoTIFF (LiDAR elevation raster).
#    cv2.IMREAD_COLOR fails on some OpenCV versions → we fall back
#    to IMREAD_ANYDEPTH and normalize to uint8 BGR so that the main
#    VO scripts (which use IMREAD_COLOR) can always load the crop.
#
#    The original full image is 729x880 with ~50% NaN (no-data) pixels.
#    We auto-detect the densest data region and crop there.
# ==============================================================
src_snow = os.path.join(out_dir, "SD_TLS_190530_1m_Red_AlignSeparadoOK.tif")
if not os.path.exists(src_snow):
    print(f"\nWARNING: Snow image not found at {src_snow} — skipping snow crop.")
else:
    print(f"\nLoading snow image: {src_snow}...")
    img_snow, normalized = load_as_bgr_uint8(src_snow)
    if img_snow is None:
        print(f"  ERROR: Could not load {src_snow}")
    else:
        H3, W3 = img_snow.shape[:2]
        print(f"  Loaded: {W3}x{H3}{' [normalized float64 -> uint8]' if normalized else ''}")

        # The image is already small (729x880) — find dense data region.
        # Use the green channel as a proxy for data presence after normalization.
        gray_snow = cv2.cvtColor(img_snow, cv2.COLOR_BGR2GRAY)
        _, data_mask = cv2.threshold(gray_snow, 1, 255, cv2.THRESH_BINARY)
        coords = cv2.findNonZero(data_mask)

        if coords is not None:
            xd, yd, wd, hd = cv2.boundingRect(coords)
            data_cx = xd + wd // 2
            data_cy = yd + hd // 2
            print(f"  Data bbox: x={xd}..{xd+wd}  y={yd}..{yd+hd}  center=({data_cx},{data_cy})")
        else:
            data_cx = W3 // 2
            data_cy = H3 // 2

        # Crop size: take as much of the image as possible without going below 300x300
        # (the camera window is 300px; we need the trajectory to fit comfortably)
        # Since the image is only 729x880, use 90% of each dimension
        CW3 = min(W3, max(W3 - 20, 300))
        CH3 = min(H3, max(H3 - 20, 300))

        x1 = max(0, data_cx - CW3 // 2)
        y1 = max(0, data_cy - CH3 // 2)
        x2 = min(W3, x1 + CW3)
        y2 = min(H3, y1 + CH3)
        x1 = max(0, x2 - CW3)
        y1 = max(0, y2 - CH3)

        crop_snow = img_snow[y1:y2, x1:x2]
        out_snow = os.path.join(out_dir, "snow_crop.tif")
        cv2.imwrite(out_snow, crop_snow)
        print(f"  Saved {out_snow}  size={crop_snow.shape[1]}x{crop_snow.shape[0]}  region=[{x1}:{x2}, {y1}:{y2}]")
        if normalized:
            print("  NOTE: Saved as normalized uint8 BGR (elevation range mapped to 0-255).")
            print("        The main VO scripts will load this with IMREAD_COLOR — compatible.")

print("\nDone. Crops ready for experiments.")
