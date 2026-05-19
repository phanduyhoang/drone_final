# PROMPT FOR AI: Thesis Visual Odometry Experiments

## What this folder is

This is a **self-contained** experiment runner for a master's thesis on aerial Visual Odometry (VO)
with Loop Closure (LC) using SuperPoint + SuperGlue. The system simulates a drone flying over an
aerial image and estimates its trajectory from visual features alone (GPS-denied navigation).

All dependencies (model code, weights, metric utilities) are already inside this folder.
No external project folder is required. Just activate the conda environment and run.

## Folder structure

```
thesis_experiments/           ← this folder
├── PROMPT.md                 ← you are reading this
├── run_experiments.bat       ← run all 6 experiments
├── crop_images.py            ← create cropped images (run once first)
├── square_loop_400steps_10px_smooth.json   ← trajectory (auto-scales to any image)
├── metrics.py                ← TrackingMetrics class
├── environment.yml           ← conda env spec (if you need to recreate the env)
│
├── magicpoint/
│   ├── supereye.py           ← SuperPoint network + frontend
│   └── superpoint_v1.pth    ← SuperPoint pretrained weights
│
├── SuperGluePretrainedNetwork-master/
│   └── models/
│       └── superglue.py      ← SuperGlue matching network
│
├── images/
│   ├── corner_crop_terrain.tif    ← 9000×8000 terrain aerial crop (RGB)
│   ├── snow_crop.tif              ← clean snow/elevation crop (uint8 BGR)
│   ├── summer_city_crop.tif       ← 9000×8000 summer city crop (RGB)
│   └── SD_TLS_190530_1m_Red_AlignSeparadoOK.tif  ← original snow image
│       (+ Corner_orthoimage.tif and summer_city.tif if you need to re-crop)
│
├── results/                  ← output PNGs go here (auto-created)
│
└── methods/
    ├── superpoint_simulation_aerial_vo_only_from_trajectory.py
    └── superpoint_simulation_aerial_with_loopclosure_posegraph_from_trajectory.py
```

## Quick start

```bat
rem Activate the conda environment first (env is named vo_thesis)
conda activate vo_thesis

cd thesis_experiments

rem Step 1: create cropped images (only needed once, or if images/ is missing crops)
python crop_images.py

rem Step 2: run all 6 experiments
run_experiments.bat
```

If `vo_thesis` env doesn't exist yet, create it from the spec:
```bat
conda env create -f environment.yml
conda activate vo_thesis
```
The environment needs: python 3.10, numpy, torch 2.6+cu124, torchvision, opencv-python.
A CUDA-capable GPU (≥4 GB VRAM) is required for reasonable speed.

The 6 experiments are:
1. VO-only on terrain image
2. VO+LC on terrain image
3. VO-only on snow image
4. VO+LC on snow image
5. VO-only on summer city image
6. VO+LC on summer city image

Each run prints metrics to stdout and saves two PNG files to `results/`:
- `*_map.png` — dark canvas: GT (red) vs estimated (green) trajectory
- `*_overlay.png` — aerial image with both trajectories drawn on it

## Critical design decisions (READ BEFORE TOUCHING ANYTHING)

### 1. `--cam-sign 1 --yaw-use prev` is MANDATORY for all runs

`cam_sign` controls how camera rotation is simulated during the experiment.
The correct value is **+1** (proven by the original thesis on big.tif: ATE 93px for VO+LC).

The scripts include an auto-calibration function that tries to detect cam_sign from the image.
This works on high-texture images but **FAILS on low-texture images** (snow, uniform terrain):
it detects cam_sign=−1 (WRONG) → spurious translation at every trajectory corner → staircase
drift pattern → trajectory hundreds of pixels off.

**Fix:** `--cam-sign 1 --yaw-use prev` forces cam_sign=1 and skips calibration entirely.
All runs in `run_experiments.bat` already use this. **Never remove it.**

### 2. Trajectory scaling is automatic — do not modify the JSON

The trajectory JSON has `"image_size": [10000, 10000]`. The scripts scale it to any image:
```python
sx = W / 10000.0
sy = H / 10000.0
```
The trajectory (a 4000×4000px square in 10000×10000 reference space) auto-scales to:
- 9000×8000 terrain: square spans ~3600×3200px
- 880×729 snow: square spans ~352×292px
- 9000×8000 city: square spans ~3600×3200px

**Do not hardcode coordinates.** The JSON works on any image size.

### 3. Snow image notes

The snow image (`SD_TLS_190530_1m_Red_AlignSeparadoOK.tif`) is a 64-bit float LiDAR elevation
raster (TLS = Terrestrial Laser Scanner). It is 880×729 pixels with ~50% NaN no-data pixels.

`crop_images.py` creates `snow_crop.tif` (860×709 uint8 BGR) by:
- Loading with `cv2.IMREAD_ANYDEPTH` (handles float64 TIFFs)
- Normalizing the elevation range (−0.5 to +7.2 m) to 0–255 grayscale
- Saving as uint8 BGR TIF so `cv2.IMREAD_COLOR` in the main scripts always succeeds

The main scripts load `snow_crop.tif` with `cv2.IMREAD_COLOR` (uint8 BGR). This is consistent
with how terrain and city images are loaded. The resulting grayscale elevation texture gives
SuperPoint enough features to track.

If `snow_crop.tif` does not exist, `run_experiments.bat` automatically falls back to the
original snow image with `--cx 0.6 --cy 0.55` (trajectory centering shift).

### 4. How the VO pipeline works (brief)

For each frame in the simulated flight:
1. Extract camera window from aerial image at (x, y) with yaw rotation applied
2. Stabilize: rotate back by (−yaw) so north is always up
3. Run SuperPoint → keypoints + 256-D descriptors
4. Run SuperGlue → match with previous frame
5. RANSAC affine → translation (dx, dy)
6. `est_position += (−dx, −dy)` (features move opposite to camera)

For VO+LC additionally:
- Store keyframes every 40px of estimated movement (max 2000 in DB)
- Every 25 frames: search for a nearby keyframe visited ≥400 frames ago
- If found with ≥80 inliers: distribute drift correction linearly from anchor→current

### 5. Metrics interpretation

| Metric | Meaning | Typical values (20 loops) |
|--------|---------|--------------------------|
| ATE | RMSE over all frames (grows with loops) | VO-only: 400–1200px; VO+LC: 50–200px |
| RPE | Per-frame average error (stable) | 5–20px per frame |
| Mean features | Avg keypoints found per frame | 100–400 |
| Mean inliers | Avg RANSAC inliers per match | 50–200 |

**Thesis baseline** (big.tif, 70 loops):
- VO-only: ATE 1003px, RPE 3.8px
- VO+LC:  ATE   93px, RPE 3.7px

The new images with 20 loops will have different absolute ATE but the VO+LC/VO-only ratio
should still show dramatic improvement, proving the system generalizes to diverse imagery.

## Troubleshooting

### "Could not import SuperGlue"
`SuperGluePretrainedNetwork-master/` is missing. Check it exists in this folder (not a parent).

### "Could not load image" / FileNotFoundError
- Run `python crop_images.py` first to create the cropped images.
- Check that `images/corner_crop_terrain.tif`, `images/snow_crop.tif`, and
  `images/summer_city_crop.tif` exist.

### Trajectory looks like a staircase / spiraling off
`cam_sign` is wrong. Verify that `--cam-sign 1 --yaw-use prev` is in the command.
Never let auto-calibration run on the snow or low-texture images.

### Estimated trajectory barely moves / straight line
SuperPoint found almost no features. Likely causes:
- Trajectory is on a black (NaN/no-data) region of the image
- For the snow image: run `python crop_images.py` to create `snow_crop.tif` and use that
- Check the overlay PNG: if the red GT line is over black image area, the trajectory is misplaced

### "ATE: nan" or metrics crash
`metrics.py` missing. It must be in the same folder as this PROMPT.md.

### GPU out-of-memory
Reduce keypoints: `--max-kp 300` or loops: `--loops 10`.

### TIFF warning: "can not handle images with 64-bit samples"
This is a benign warning from `crop_images.py` when loading the float64 snow TIFF.
It is handled automatically — `crop_images.py` falls back to `IMREAD_ANYDEPTH` and
normalizes to uint8. The warning does not prevent the crop from being saved.

## Running a single experiment manually

```bat
cd thesis_experiments

python methods\superpoint_simulation_aerial_with_loopclosure_posegraph_from_trajectory.py ^
  --image images\corner_crop_terrain.tif ^
  --trajectory square_loop_400steps_10px_smooth.json ^
  --cam-sign 1 --yaw-use prev --loops 20 ^
  --save-traj-map results\test_lc_map.png ^
  --save-gt-overlay results\test_lc_overlay.png
```

For the **snow image**, add `--cx 0.6 --cy 0.55` if you are using the original
`SD_TLS_190530_1m_Red_AlignSeparadoOK.tif`. For `snow_crop.tif`, no extra flags needed.

## File reference

| File | Purpose |
|------|---------|
| `run_experiments.bat` | Runs all 6 experiments in sequence |
| `crop_images.py` | Creates cropped images (run once before experiments) |
| `square_loop_400steps_10px_smooth.json` | Square trajectory, auto-scales to any image |
| `metrics.py` | TrackingMetrics: ATE, RPE, feature stats |
| `magicpoint/supereye.py` | SuperPoint network class + inference frontend |
| `magicpoint/superpoint_v1.pth` | SuperPoint pretrained weights |
| `SuperGluePretrainedNetwork-master/` | SuperGlue matching network |
| `methods/..._vo_only....py` | VO-only pipeline (no loop closure) |
| `methods/..._with_loopclosure....py` | VO + Loop Closure pipeline |
| `images/corner_crop_terrain.tif` | 9000×8000 real aerial terrain (RGB) |
| `images/snow_crop.tif` | 860×709 elevation raster (normalized to uint8) |
| `images/summer_city_crop.tif` | 9000×8000 aerial city summer (RGB) |
| `results/` | Output directory (auto-created by run_experiments.bat) |
