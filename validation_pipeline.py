# ============================================================
# GEOMETRIC VALIDATION PIPELINE
# ============================================================
# Full validation workflow for imaging consistency analysis
#
# MODULE 1 : Interactive Annotation Engine (OpenCV)
# MODULE 2 : Analytics + Statistics + Visualization
#
# Features:
# - Manual two-point measurement (Length + Width)
# - Euclidean distance calculation
# - Spatial scale analysis (mm/px)
# - Aspect ratio analysis
# - CV calculation
# - Publication-quality plots
# - CSV + Excel export
#
# Author: Reconstructed for phenotypic validation workflow
# ============================================================

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import random
import math

# ============================================================
# CONFIGURATION
# ============================================================

# Directory containing raw images
IMAGE_DIR = "raw_images"

# Output directory
OUTPUT_DIR = "validation_results"

# Known physical size of reference patch (mm)
REFERENCE_MM = 14.0

# Number of images to sample
N_IMAGES = 100

# Supported image formats
IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png", ".tif", ".tiff"]

# ============================================================
# CREATE OUTPUT DIR
# ============================================================

output_dir = Path(OUTPUT_DIR)
output_dir.mkdir(parents=True, exist_ok=True)

# ============================================================
# GLOBAL VARIABLES FOR INTERACTION
# ============================================================

clicked_points = []
current_image = None
display_image = None

# ============================================================
# MOUSE CALLBACK
# ============================================================

def mouse_callback(event, x, y, flags, param):

    global clicked_points, display_image

    if event == cv2.EVENT_LBUTTONDOWN:

        clicked_points.append((x, y))

        # Draw first point
        if len(clicked_points) == 1:
            cv2.circle(display_image, (x, y), 6, (0, 0, 255), -1)

        # Draw second point and line
        elif len(clicked_points) == 2:
            cv2.circle(display_image, (x, y), 6, (0, 255, 0), -1)

            cv2.line(
                display_image,
                clicked_points[0],
                clicked_points[1],
                (0, 255, 0),
                2
            )

        cv2.imshow("Measurement Window", display_image)

# ============================================================
# EUCLIDEAN DISTANCE
# ============================================================

def euclidean_distance(p1, p2):

    return math.sqrt(
        (p2[0] - p1[0])**2 +
        (p2[1] - p1[1])**2
    )

# ============================================================
# INTERACTIVE MEASUREMENT
# ============================================================

def measure_dimension(image, title="Measure"):

    global clicked_points, display_image

    clicked_points = []

    display_image = image.copy()

    cv2.namedWindow("Measurement Window", cv2.WINDOW_NORMAL)
    cv2.imshow("Measurement Window", display_image)

    cv2.setMouseCallback("Measurement Window", mouse_callback)

    print("\n================================================")
    print(title)
    print("1. Click START point")
    print("2. Click END point")
    print("3. Press ENTER to confirm")
    print("4. Press R to reset")
    print("5. Press ESC to skip")
    print("================================================")

    while True:

        key = cv2.waitKey(1) & 0xFF

        # ENTER
        if key == 13:

            if len(clicked_points) == 2:

                distance = euclidean_distance(
                    clicked_points[0],
                    clicked_points[1]
                )

                return distance, clicked_points

        # RESET
        elif key == ord('r'):

            clicked_points = []
            display_image = image.copy()
            cv2.imshow("Measurement Window", display_image)

        # ESC
        elif key == 27:

            return None, None

# ============================================================
# IMAGE SAMPLING
# ============================================================

def load_sample_images(image_dir, n=100):

    image_dir = Path(image_dir)

    image_paths = []

    for ext in IMAGE_EXTENSIONS:
        image_paths.extend(image_dir.glob(f"*{ext}"))

    image_paths = sorted(image_paths)

    if len(image_paths) == 0:
        raise Exception(f"No images found in {image_dir}")

    if len(image_paths) < n:
        print(f"WARNING: Only {len(image_paths)} images found.")
        n = len(image_paths)

    sampled = random.sample(image_paths, n)

    return sampled

# ============================================================
# MAIN INTERACTIVE ANNOTATION MODULE
# ============================================================

def interactive_annotation():

    print("\nLoading images...")

    sampled_images = load_sample_images(IMAGE_DIR, N_IMAGES)

    results = []

    for idx, image_path in enumerate(sampled_images):

        print(f"\n[{idx+1}/{len(sampled_images)}]")
        print(f"Image: {image_path.name}")

        image = cv2.imread(str(image_path))

        if image is None:
            print("Failed to load image.")
            continue

        # Resize for viewing if needed
        max_width = 1400

        if image.shape[1] > max_width:

            scale = max_width / image.shape[1]

            image = cv2.resize(
                image,
                None,
                fx=scale,
                fy=scale
            )

        # ----------------------------------------------------
        # LENGTH MEASUREMENT
        # ----------------------------------------------------

        length_px, length_points = measure_dimension(
            image,
            title="Measure PATCH LENGTH"
        )

        if length_px is None:
            print("Skipped.")
            continue

        # ----------------------------------------------------
        # WIDTH MEASUREMENT
        # ----------------------------------------------------

        width_px, width_points = measure_dimension(
            image,
            title="Measure PATCH WIDTH"
        )

        if width_px is None:
            print("Skipped.")
            continue

        # ----------------------------------------------------
        # CALCULATIONS
        # ----------------------------------------------------

        # mean_pixels = (length_px + width_px) / 2
        # mm_per_px = REFERENCE_MM / mean_pixels
        
        scale_length = REFERENCE_MM / length_px
        scale_width  = REFERENCE_MM / width_px
        mm_per_px = (scale_length + scale_width) / 2

        aspect_ratio = length_px / width_px

        results.append({

            "image_name": image_path.name,

            "length_px": length_px,
            "width_px": width_px,

            "length_x1": length_points[0][0],
            "length_y1": length_points[0][1],
            "length_x2": length_points[1][0],
            "length_y2": length_points[1][1],

            "width_x1": width_points[0][0],
            "width_y1": width_points[0][1],
            "width_x2": width_points[1][0],
            "width_y2": width_points[1][1],

            "mean_pixels": mean_pixels,

            "mm_per_px": mm_per_px,

            "aspect_ratio": aspect_ratio

        })

        print(f"Length (px): {length_px:.2f}")
        print(f"Width  (px): {width_px:.2f}")
        print(f"mm/px       : {mm_per_px:.5f}")
        print(f"AspectRatio : {aspect_ratio:.4f}")

    cv2.destroyAllWindows()

    return pd.DataFrame(results)

# ============================================================
# ANALYTICS + VISUALIZATION MODULE
# ============================================================

def analytics_and_plots(df):

    print("\n================================================")
    print("RUNNING ANALYTICS")
    print("================================================")

    # --------------------------------------------------------
    # SPATIAL SCALE STATS
    # --------------------------------------------------------

    scale_mean = df["mm_per_px"].mean()

    scale_std = df["mm_per_px"].std()

    scale_cv = (scale_std / scale_mean) * 100

    # --------------------------------------------------------
    # ASPECT RATIO STATS
    # --------------------------------------------------------

    ar_mean = df["aspect_ratio"].mean()

    ar_std = df["aspect_ratio"].std()

    # --------------------------------------------------------
    # PRINT SUMMARY
    # --------------------------------------------------------

    print("\nSpatial Scale Analysis")
    print("-----------------------------------")
    print(f"Mean mm/px      : {scale_mean:.4f}")
    print(f"Std mm/px       : {scale_std:.4f}")
    print(f"CV (%)          : {scale_cv:.2f}")

    print("\nAspect Ratio Analysis")
    print("-----------------------------------")
    print(f"Mean AR         : {ar_mean:.4f}")
    print(f"Std AR          : {ar_std:.4f}")

    # --------------------------------------------------------
    # SAVE SUMMARY
    # --------------------------------------------------------

    summary_df = pd.DataFrame({

        "metric": [
            "mean_mm_per_px",
            "std_mm_per_px",
            "cv_percent",
            "mean_aspect_ratio",
            "std_aspect_ratio"
        ],

        "value": [
            scale_mean,
            scale_std,
            scale_cv,
            ar_mean,
            ar_std
        ]
    })

    summary_path = output_dir / "summary_statistics.xlsx"

    summary_df.to_excel(summary_path, index=False)

    # --------------------------------------------------------
    # SAVE RAW DATA
    # --------------------------------------------------------

    csv_path = output_dir / "validation_measurements.csv"

    xlsx_path = output_dir / "validation_measurements.xlsx"

    df.to_csv(csv_path, index=False)

    df.to_excel(xlsx_path, index=False)

    print(f"\nSaved:")
    print(csv_path)
    print(xlsx_path)
    print(summary_path)

    # ========================================================
    # PLOT 1 : SPATIAL SCALE CONSISTENCY
    # ========================================================

    plt.figure(figsize=(8, 6))

    plt.boxplot(
        df["mm_per_px"],
        vert=True
    )

    jitter_x = np.random.normal(1, 0.03, len(df))

    plt.scatter(
        jitter_x,
        df["mm_per_px"],
        alpha=0.7
    )

    plt.ylabel("Spatial Scale (mm/px)")
    plt.title("Spatial Scale Consistency")

    text = (
        f"Mean = {scale_mean:.4f} mm/px\n"
        f"CV = {scale_cv:.2f}%"
    )

    plt.text(
        1.15,
        scale_mean,
        text
    )

    plt.tight_layout()

    scale_plot_path = output_dir / "spatial_scale_consistency.png"

    plt.savefig(scale_plot_path, dpi=300)

    plt.close()

    # ========================================================
    # PLOT 2 : GEOMETRIC SHAPE INTEGRITY
    # ========================================================

    plt.figure(figsize=(7, 7))

    plt.scatter(
        df["width_px"],
        df["length_px"],
        alpha=0.8
    )

    # Ideal line
    min_val = min(
        df["width_px"].min(),
        df["length_px"].min()
    )

    max_val = max(
        df["width_px"].max(),
        df["length_px"].max()
    )

    plt.plot(
        [min_val, max_val],
        [min_val, max_val],
        linestyle="--"
    )

    plt.xlabel("Width (px)")
    plt.ylabel("Length (px)")
    plt.title("Geometric Shape Integrity")

    text = (
        f"Mean AR = {ar_mean:.3f}\n"
        f"Std = {ar_std:.4f}"
    )

    plt.text(
        min_val,
        max_val,
        text
    )

    plt.tight_layout()

    shape_plot_path = output_dir / "geometric_shape_integrity.png"

    plt.savefig(shape_plot_path, dpi=300)

    plt.close()

    print("\nPlots saved:")
    print(scale_plot_path)
    print(shape_plot_path)

# ============================================================
# MAIN
# ============================================================

def main():

    print("\n================================================")
    print("GEOMETRIC VALIDATION PIPELINE")
    print("================================================")

    df = interactive_annotation()

    if len(df) == 0:
        print("No measurements collected.")
        return

    analytics_and_plots(df)

    print("\n================================================")
    print("PIPELINE COMPLETE")
    print("================================================")

if __name__ == "__main__":
    main()