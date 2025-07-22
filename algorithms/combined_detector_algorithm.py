#!/usr/bin/env python3
"""
Script Name: combined_detector_algorithm.py

Summary:
This script demonstrates how to open Whole Slide Images (WSIs) with OpenSlide, extract a thumbnail
at level 2 (or level 1 if level 2 doesn't exist), and then apply a combined tissue-detection
approach (using FilterGrays + DownsampleKMeansDetector). The resulting binary mask (0 = background, 255 = tissue)
is saved for each processed slide. A CSV file is also generated to report the processing time
per WSI.

Usage:
    python combined_detector_algorithm.py

Notes:
1. We replicate the "open WSI -> pick level 2 -> read region -> convert to numpy" logic
   from your reference K-Means script.
2. We keep the DoublePass approach, which uses both FilterGrays and Bogdan2 to create a
   tissue mask.
3. If a slide has only 0-level, we skip it (same logic as your reference).
4. The mask is saved in the directory "0_255" as a PNG file with 0/255 intensities.
"""

import os
import csv
import time
import warnings
import cv2
import numpy as np
import openslide
import skimage.filters as sk_filters
import skimage.morphology as sk_morphology
import matplotlib.pyplot as plt
import abc
from tqdm import tqdm
from typing import Optional, Tuple
from get_thumbnail import generate_thumbnail
# --------------------- TissueDetector Base Class -----------------------------
class TissueDetector(abc.ABC):
    def __init__(self) -> None:
        self.mask: Optional[np.ndarray] = None

    @abc.abstractmethod
    def detect_tissue(self, thumbnail: np.ndarray) -> None:
        """
        Subclasses must implement detect_tissue() to fill self.mask (binary).
        """
        pass

    def visualize_results(self, thumbnail: np.ndarray, truth: Optional[np.ndarray] = None) -> None:
        """
        Visualize the thumbnail, the generated tissue mask, and optionally ground truth.
        """
        fig, axs = plt.subplots(1, 3 if truth is not None else 2, figsize=(18, 6))
        axs[0].imshow(cv2.cvtColor(thumbnail, cv2.COLOR_BGR2RGB))
        axs[0].set_title("Thumbnail")
        axs[0].axis('off')
        
        axs[1].imshow(self.mask, cmap='gray')
        axs[1].set_title("Tissue Mask")
        axs[1].axis('off')
        
        if truth is not None:
            axs[2].imshow(truth, cmap='gray')
            axs[2].set_title("Ground Truth")
            axs[2].axis('off')
        
        plt.show()

# --------------------- FilterGrays Detector -----------------------------
class FilterGrays(TissueDetector):
    def __init__(self) -> None:
        super().__init__()

    def detect_tissue(self, thumbnail: np.ndarray) -> None:
        """
        Detect tissue by filtering out gray regions, morphological operations,
        and small-object removal.
        """
        # Sharpen
        kernel = np.array([[0, -1, 0],
                           [-1, 5, -1],
                           [0, -1, 0]])
        sharpened_image = cv2.filter2D(thumbnail, -1, kernel)
        self.mask = self.filter_grays(sharpened_image, tolerance=15, output_type="uint8")
        
        # Morphological filtering to reduce noise
        kernel = np.ones((5, 5), np.uint8)
        dilated_image = cv2.dilate(self.mask, kernel, iterations=1)
        cleaned_mask = cv2.morphologyEx(dilated_image, cv2.MORPH_CLOSE, kernel)
        
        # Remove small objects
        cleaned_mask = sk_morphology.remove_small_objects(cleaned_mask.astype(bool), min_size=5000)
        self.mask = cleaned_mask.astype(np.uint8)  # 0 or 1
        return self.mask

    def filter_grays(self, rgb: np.ndarray, tolerance: int = 15, output_type: str = "bool") -> np.ndarray:
        """
        Filter out pixels that are roughly gray (R ~ G ~ B) within a given tolerance.
        """
        (h, w, c) = rgb.shape
        rgb_int = rgb.astype(int)
        rg_diff = np.abs(rgb_int[:, :, 0] - rgb_int[:, :, 1]) <= tolerance
        rb_diff = np.abs(rgb_int[:, :, 0] - rgb_int[:, :, 2]) <= tolerance
        gb_diff = np.abs(rgb_int[:, :, 1] - rgb_int[:, :, 2]) <= tolerance
        result = ~(rg_diff & rb_diff & gb_diff)
        
        if output_type == "bool":
            return result
        elif output_type == "float":
            return result.astype(float)
        else:  # "uint8"
            return result.astype("uint8")

# --------------------- DownsampleKMeansDetector Detector -----------------------------
class DownsampleKMeansDetector(TissueDetector):
    def __init__(self) -> None:
        super().__init__()

    def detect_tissue(self, thumbnail: np.ndarray) -> None:
        """
        Detect tissue by:
        1. Resizing the image smaller
        2. K-means with 2 clusters
        3. Picking the cluster that is presumably the tissue
        4. Upscaling the mask and morphological cleanup
        """
        # Downscale for K-Means
        small_image = cv2.resize(
            thumbnail,
            (thumbnail.shape[1] // 4, thumbnail.shape[0] // 4),
            interpolation=cv2.INTER_LINEAR
        )

        # Prepare data for K-Means
        pixel_values = small_image.reshape((-1, 3))
        pixel_values = np.float32(pixel_values)

        # Apply K-Means
        k = 2
        _, labels, centers = cv2.kmeans(
            data=pixel_values,
            K=k,
            bestLabels=None,
            criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2),
            attempts=10,
            flags=cv2.KMEANS_RANDOM_CENTERS
        )

        # Rebuild segmented image (optional step)
        centers = np.uint8(centers)
        labels = labels.flatten()
        # segmented_image = centers[labels.flatten()].reshape(small_image.shape)  # unused in mask

        # Determine which cluster is tissue
        # We'll pick the cluster with the *lowest mean intensity* (like dark tissues)
        # Alternatively, you can pick the cluster with the highest mean intensity, depending on data
        tissue_cluster = np.argmin(np.mean(centers, axis=1))

        # Create a small binary mask
        mask_small = np.zeros_like(labels, dtype=np.uint8)
        mask_small[labels == tissue_cluster] = 1
        mask_small = mask_small.reshape(small_image.shape[:2])

        # Upscale the mask to original size
        self.mask = cv2.resize(mask_small,
                               (thumbnail.shape[1], thumbnail.shape[0]),
                               interpolation=cv2.INTER_NEAREST)

        # Morphological cleanup
        kernel = np.ones((5, 5), np.uint8)
        self.mask = cv2.morphologyEx(self.mask, cv2.MORPH_CLOSE, kernel)
        self.mask = cv2.morphologyEx(self.mask, cv2.MORPH_OPEN, kernel)

        return self.mask

# --------------------- DoublePass -----------------------------
class DoublePass(TissueDetector):
    def __init__(self) -> None:
        super().__init__()

    def detect_tissue(self, thumbnail: np.ndarray) -> None:
        """
        Combine the FilterGrays and DownsampleKMeansDetector detection masks (logical OR).
        """
        fg_detector = FilterGrays()
        fg_detector.detect_tissue(thumbnail)
        mask_fg = fg_detector.mask

        bogdan2_detector = DownsampleKMeansDetector()
        bogdan2_detector.detect_tissue(thumbnail)
        mask_bogdan2 = bogdan2_detector.mask

        combined_mask = cv2.bitwise_or(mask_fg, mask_bogdan2)

        # Optional morphological cleanup
        kernel = np.ones((5, 5), np.uint8)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)

        self.mask = combined_mask
        return (self.mask > 0).astype(np.uint8) * 255

# --------------------- WSI Processing Function -----------------------------
def process_wsi(
    input_path: str,
    output_dir: str,
    detector: TissueDetector,
    target_mpp: float = 10.0
) -> Tuple[str, float]:
    """
    Generate a thumbnail using generate_thumbnail, run a TissueDetector, and save the mask.

    Args:
        input_path (str): Path to the WSI file.
        output_dir (str): Directory to save the mask.
        detector (TissueDetector): Detector instance to generate the mask.
        target_mpp (float): Target microns per pixel for thumbnail generation (default: 10.0).

    Returns:
        Tuple[str, float]: (mask_output_path, elapsed_time) if successful, ("", 0.0) if skipped.
    """
    # Generate the thumbnail
    thumbnail = generate_thumbnail(input_path, target_mpp=target_mpp)
    if thumbnail is None:
        return "", 0.0  # Skip if thumbnail generation failed
    
    # Detect tissue
    start_time = time.time()
    detector.detect_tissue(thumbnail)
    mask = detector.mask #(detector.mask > 0).astype(np.uint8) * 255
    elapsed_time = time.time() - start_time

    # Save the mask
    file_base = os.path.splitext(os.path.basename(input_path))[0]
    mask_output_path = os.path.join(output_dir, f"{file_base}.png")
    cv2.imwrite(mask_output_path, mask)

    return mask_output_path, elapsed_time

# --------------------- Main Script -----------------------------
if __name__ == "__main__":
    wsi_dir = "/home/bogdan/indonezia/data/GRAND-QC/data/wsi"
    out_dir = "/home/bogdan/indonezia/data/GRAND-QC/data/inference/combined_detector_algorithm"

    # Choose your combined detector
    combined_detector = DoublePass()

    # Process each cohort directory
    for cohort_name in tqdm(os.listdir(wsi_dir)):
        cohort_path = os.path.join(wsi_dir, cohort_name)
        
        # Skip if not a directory
        if not os.path.isdir(cohort_path):
            continue

        print(f"\nProcessing cohort: {cohort_name}")
        
        # Set up output directories for this cohort
        out_results_dir = os.path.join(out_dir, cohort_name)
        mask_output_dir = os.path.join(out_results_dir, "0_255")

        # Ensure the output directory exists
        os.makedirs(mask_output_dir, exist_ok=True)

        processing_times = []

        # Process all .svs, .tiff, or .tif in the cohort directory
        wsi_files = [f for f in os.listdir(cohort_path) if f.lower().endswith(('.svs', '.tiff', '.tif'))]
        for file_name in tqdm(wsi_files, desc=f"Processing {cohort_name} WSIs"):
            wsi_path = os.path.join(cohort_path, file_name)
            try:
                mask_path, elapsed = process_wsi(
                    input_path=wsi_path,
                    output_dir=mask_output_dir,
                    detector=combined_detector,
                    target_mpp=10.0  # Explicitly set to match dataset creation
                )
                if mask_path:  # Means we didn't skip
                    processing_times.append((file_name, elapsed))
                    print(f"Processed {file_name} in {elapsed:.2f} s. Saved mask to {mask_path}.")
                else:
                    print(f"Skipped {file_name}.")
            except Exception as e:
                print(f"Error processing {file_name}: {e}")

        # Save processing times for this cohort
        if processing_times:
            timing_csv_path = os.path.join(out_results_dir, "combined_detector_timing.csv")
            with open(timing_csv_path, mode='w', newline='') as csv_file:
                writer = csv.writer(csv_file)
                writer.writerow(["Filename", "Processing Time (seconds)"])
                writer.writerows(processing_times)

            print(f"Processing times for {cohort_name} saved to {timing_csv_path}")