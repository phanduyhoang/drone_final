@echo off
echo ===============================================
echo  Thesis VO Experiments
echo ===============================================

set PYTHON=python
set METHODS=methods
set TRAJ=square_loop_400steps_10px_smooth.json
set LOOPS=20

rem === Images ===
set IMG_TERRAIN=images\corner_crop_terrain.tif
set IMG_SNOW_CROP=images\snow_crop.tif
set IMG_SNOW_ORIG=images\SD_TLS_190530_1m_Red_AlignSeparadoOK.tif
set IMG_CITY=images\summer_city_crop.tif

rem === IMPORTANT: --cam-sign 1 --yaw-use prev skips auto-calibration entirely.
rem     This is required for reliable results on all images.
rem     The trajectory auto-scales to whatever image size you load.
set SHARED=--cam-sign 1 --yaw-use prev --loops %LOOPS%

rem === Make sure results folder exists ===
if not exist results mkdir results

rem === Check which snow image to use ===
rem     Prefer snow_crop.tif (clean crop, no --cx/--cy needed).
rem     Fall back to original snow image with --cx/--cy if crop doesn't exist.
if exist %IMG_SNOW_CROP% (
    echo Using cropped snow image: %IMG_SNOW_CROP%
    set IMG_SNOW=%IMG_SNOW_CROP%
    set SNOW_EXTRA=
) else (
    echo snow_crop.tif not found, using original snow image with --cx 0.6 --cy 0.55
    echo TIP: Run "python crop_images.py" first to create snow_crop.tif
    set IMG_SNOW=%IMG_SNOW_ORIG%
    set SNOW_EXTRA=--cx 0.6 --cy 0.55
)

echo.
echo [1/6] VO-only on Terrain (Corner crop)...
%PYTHON% %METHODS%\superpoint_simulation_aerial_vo_only_from_trajectory.py ^
  --image %IMG_TERRAIN% --trajectory %TRAJ% %SHARED% ^
  --cx 0.4 --cy 0.5 ^
  --save-traj-map results\terrain_vo_map.png --save-gt-overlay results\terrain_vo_overlay.png
echo Done.

echo.
echo [2/6] VO+LC on Terrain (Corner crop)...
%PYTHON% %METHODS%\superpoint_simulation_aerial_with_loopclosure_posegraph_from_trajectory.py ^
  --image %IMG_TERRAIN% --trajectory %TRAJ% %SHARED% ^
  --cx 0.4 --cy 0.5 ^
  --save-traj-map results\terrain_lc_map.png --save-gt-overlay results\terrain_lc_overlay.png
echo Done.

echo.
echo [3/6] VO-only on Snow image...
%PYTHON% %METHODS%\superpoint_simulation_aerial_vo_only_from_trajectory.py ^
  --image %IMG_SNOW% --trajectory %TRAJ% %SHARED% %SNOW_EXTRA% ^
  --save-traj-map results\snow_vo_map.png --save-gt-overlay results\snow_vo_overlay.png
echo Done.

echo.
echo [4/6] VO+LC on Snow image...
%PYTHON% %METHODS%\superpoint_simulation_aerial_with_loopclosure_posegraph_from_trajectory.py ^
  --image %IMG_SNOW% --trajectory %TRAJ% %SHARED% %SNOW_EXTRA% ^
  --save-traj-map results\snow_lc_map.png --save-gt-overlay results\snow_lc_overlay.png
echo Done.

echo.
echo [5/6] VO-only on Summer City...
%PYTHON% %METHODS%\superpoint_simulation_aerial_vo_only_from_trajectory.py ^
  --image %IMG_CITY% --trajectory %TRAJ% %SHARED% ^
  --save-traj-map results\city_vo_map.png --save-gt-overlay results\city_vo_overlay.png
echo Done.

echo.
echo [6/6] VO+LC on Summer City...
%PYTHON% %METHODS%\superpoint_simulation_aerial_with_loopclosure_posegraph_from_trajectory.py ^
  --image %IMG_CITY% --trajectory %TRAJ% %SHARED% ^
  --save-traj-map results\city_lc_map.png --save-gt-overlay results\city_lc_overlay.png
echo Done.

echo.
echo ===============================================
echo  All experiments finished. Results in: results\
echo ===============================================
pause
