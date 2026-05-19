#!/usr/bin/env python3
"""
run_parallel.py  --  Run VO and LC experiments in parallel.

For each image pair, VO-only and VO+LC launch simultaneously on the GPU.
Output from both processes is streamed live with prefixes so you can
follow both at once. When all pairs finish a combined metrics table is printed.

Usage:
    python run_parallel.py                  # terrain + city 70 loops, snow 20 loops
    python run_parallel.py --loops 20       # quick test
    python run_parallel.py --loops-snow 5   # even shorter snow challenge

Images:
    Terrain  : images/corner_crop_terrain.tif
    City     : images/summer_city_crop.tif
    Snow     : images/snow_crop.tif  (fallback: SD_TLS...tif + --cx 0.6 --cy 0.55)
"""

import subprocess
import threading
import os
import sys
import json
import argparse

# ---------------------------------------------------------------------------
PYTHON   = sys.executable
VO_SCRIPT = os.path.join("methods",
    "superpoint_simulation_aerial_vo_only_from_trajectory.py")
LC_SCRIPT = os.path.join("methods",
    "superpoint_simulation_aerial_with_loopclosure_posegraph_from_trajectory.py")
TRAJ     = "square_loop_400steps_10px_smooth.json"
SHARED   = ["--cam-sign", "1", "--yaw-use", "prev"]
# ---------------------------------------------------------------------------


def stream(proc, prefix):
    """Forward subprocess stdout line-by-line with a label prefix."""
    for line in iter(proc.stdout.readline, b""):
        print(f"[{prefix}] {line.decode('utf-8', errors='replace').rstrip()}",
              flush=True)


def run_pair(label_vo, args_vo, label_lc, args_lc):
    """Launch VO and LC simultaneously, stream both, wait for both."""
    print(f"\n{'='*68}")
    print(f"  PARALLEL: {label_vo}  +  {label_lc}")
    print(f"{'='*68}\n")

    p_vo = subprocess.Popen([PYTHON] + args_vo,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    p_lc = subprocess.Popen([PYTHON] + args_lc,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

    t_vo = threading.Thread(target=stream, args=(p_vo, label_vo), daemon=True)
    t_lc = threading.Thread(target=stream, args=(p_lc, label_lc), daemon=True)
    t_vo.start()
    t_lc.start()

    p_vo.wait()
    p_lc.wait()
    t_vo.join()
    t_lc.join()

    ok_vo = "OK" if p_vo.returncode == 0 else f"ERR({p_vo.returncode})"
    ok_lc = "OK" if p_lc.returncode == 0 else f"ERR({p_lc.returncode})"
    print(f"\n[DONE] {label_vo}: {ok_vo}  |  {label_lc}: {ok_lc}")
    return p_vo.returncode, p_lc.returncode


def build_args(script, image, loops, cx=None, cy=None, extra_flags=None,
               save_map=None, save_overlay=None, save_metrics=None):
    """Build argument list for a single script invocation."""
    a = [script, "--image", image, "--trajectory", TRAJ] + SHARED + \
        ["--loops", str(loops)]
    if cx is not None:
        a += ["--cx", str(cx)]
    if cy is not None:
        a += ["--cy", str(cy)]
    if extra_flags:
        a += extra_flags
    if save_map:
        a += ["--save-traj-map", save_map]
    if save_overlay:
        a += ["--save-gt-overlay", save_overlay]
    if save_metrics:
        a += ["--save-metrics", save_metrics]
    return a


def load_metrics(path):
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None


def print_summary(rows):
    """Print the final comparison table."""
    print("\n" + "="*72)
    print("  RESULTS SUMMARY")
    print("="*72)
    hdr = f"{'Experiment':<38} {'ATE (px)':>9} {'RPE (px)':>9} {'Feats':>7} {'Inliers':>8}"
    print(hdr)
    print("-"*72)
    for label, path in rows:
        m = load_metrics(path)
        if m:
            print(f"{label:<38} {m['ATE_px']:>9.1f} {m['RPE_px']:>9.2f} "
                  f"{m['mean_features']:>7.0f} {m['mean_inlier_ratio_pct']:>7.1f}%")
        else:
            print(f"{label:<38} {'N/A':>9} {'N/A':>9} {'N/A':>7} {'N/A':>8}")
    print("="*72)
    print()


def main():
    ap = argparse.ArgumentParser(description="Parallel VO / VO+LC experiment runner")
    ap.add_argument("--loops",      type=int, default=70,
                    help="Loops for terrain and city (default 70)")
    ap.add_argument("--loops-snow", type=int, default=20,
                    help="Loops for snow challenge case (default 20)")
    args = ap.parse_args()

    os.makedirs("results", exist_ok=True)

    # Resolve snow image
    snow_img   = "images/snow_crop.tif"
    snow_extra = []
    if not os.path.exists(snow_img):
        snow_img   = "images/SD_TLS_190530_1m_Red_AlignSeparadoOK.tif"
        snow_extra = ["--cx", "0.6", "--cy", "0.55"]
        print(f"[INFO] snow_crop.tif not found, using original: {snow_img}")
    else:
        print(f"[INFO] Snow image: {snow_img}")

    summary_rows = []

    # ------------------------------------------------------------------ #
    #  1. Terrain  (cx=0.4, cy=0.3 — shift left only, y proven good)     #
    # ------------------------------------------------------------------ #
    print(f"\n[TERRAIN] {args.loops} loops")
    vo = build_args(VO_SCRIPT, "images/corner_crop_terrain.tif", args.loops,
                    cx=0.4, cy=0.3,
                    save_map="results/terrain_vo_map.png",
                    save_overlay="results/terrain_vo_overlay.png",
                    save_metrics="results/terrain_vo_metrics.json")
    lc = build_args(LC_SCRIPT, "images/corner_crop_terrain.tif", args.loops,
                    cx=0.4, cy=0.3,
                    save_map="results/terrain_lc_map.png",
                    save_overlay="results/terrain_lc_overlay.png",
                    save_metrics="results/terrain_lc_metrics.json")
    run_pair("Terrain  VO-only", vo, "Terrain  VO+LC", lc)
    summary_rows += [("Terrain  VO-only",  "results/terrain_vo_metrics.json"),
                     ("Terrain  VO+LC",    "results/terrain_lc_metrics.json")]

    # ------------------------------------------------------------------ #
    #  2. City                                                            #
    # ------------------------------------------------------------------ #
    print(f"\n[CITY] {args.loops} loops")
    vo = build_args(VO_SCRIPT, "images/summer_city_crop.tif", args.loops,
                    save_map="results/city_vo_map.png",
                    save_overlay="results/city_vo_overlay.png",
                    save_metrics="results/city_vo_metrics.json")
    lc = build_args(LC_SCRIPT, "images/summer_city_crop.tif", args.loops,
                    save_map="results/city_lc_map.png",
                    save_overlay="results/city_lc_overlay.png",
                    save_metrics="results/city_lc_metrics.json")
    run_pair("City     VO-only", vo, "City     VO+LC", lc)
    summary_rows += [("City     VO-only",  "results/city_vo_metrics.json"),
                     ("City     VO+LC",    "results/city_lc_metrics.json")]

    # ------------------------------------------------------------------ #
    #  3. Snow — challenge case                                           #
    # ------------------------------------------------------------------ #
    print(f"\n[SNOW] {args.loops_snow} loops (challenge case)")
    vo = build_args(VO_SCRIPT, snow_img, args.loops_snow,
                    extra_flags=snow_extra,
                    save_map="results/snow_vo_map.png",
                    save_overlay="results/snow_vo_overlay.png",
                    save_metrics="results/snow_vo_metrics.json")
    lc = build_args(LC_SCRIPT, snow_img, args.loops_snow,
                    extra_flags=snow_extra,
                    save_map="results/snow_lc_map.png",
                    save_overlay="results/snow_lc_overlay.png",
                    save_metrics="results/snow_lc_metrics.json")
    run_pair("Snow     VO-only (challenge)", vo, "Snow     VO+LC (challenge)", lc)
    summary_rows += [("Snow     VO-only (challenge)", "results/snow_vo_metrics.json"),
                     ("Snow     VO+LC  (challenge)",  "results/snow_lc_metrics.json")]

    # ------------------------------------------------------------------ #
    #  Final table                                                        #
    # ------------------------------------------------------------------ #
    print_summary(summary_rows)
    print("All results saved to results/")


if __name__ == "__main__":
    main()
