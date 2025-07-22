import os
import csv
import time
import cv2
import numpy as np
from tqdm import tqdm
import openslide
from sklearn.cluster import KMeans
from get_thumbnail import generate_thumbnail
from typing import Optional


class KmeansTissueDetector:
    def __init__(self, n_clusters=2):
        self.n_clusters = n_clusters
        self.mask: Optional[np.ndarray] = None

    def detect_tissue(self, thumbnail):
        height, width, _ = thumbnail.shape

        # Flatten thumbnail and compute statistics
        flat_pixels = thumbnail.reshape((-1, 3)).astype(np.float32) / 255.0
        rgb_mean = flat_pixels.mean(axis=1)
        rgb_std = flat_pixels.std(axis=1)
        features = np.stack([rgb_mean, rgb_std], axis=1)

        # Apply K-means clustering
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=0).fit(features)
        labels = kmeans.labels_.reshape(height, width)

        # Determine non-tissue cluster based on mean intensity
        cluster_means = [features[kmeans.labels_ == i][:, 0].mean() for i in range(self.n_clusters)]
        non_tissue_cluster = np.argmin(cluster_means)

        # Create binary mask with tissue as 255 and background as 0
        self.mask = (labels == non_tissue_cluster).astype(np.uint8) * 255

        # Post-processing
        kernel = np.ones((5, 5), np.uint8)
        self.mask = cv2.morphologyEx(self.mask, cv2.MORPH_CLOSE, kernel)
        self.mask = cv2.morphologyEx(self.mask, cv2.MORPH_OPEN, kernel)

        #Store and return the mask
        return self.mask


def process_wsi(input_path, mask_output_dir, detector, target_mpp=10.0):
    """
    Open the WSI with OpenSlide, generate a thumbnail at the specified target mpp,
    run the TissueDetector, and save the resulting mask as a 0/255 PNG.

    Args:
        input_path (str): Path to the WSI file.
        mask_output_dir (str): Directory to save the mask.
        detector (TissueDetector): Detector instance to generate the mask.
        target_mpp (float): Target microns per pixel for thumbnail generation (default: 10.0).

    Returns:
        tuple: (mask_output_path, detection_time) if successful, (None, None) if skipped.
    """
    thumbnail = generate_thumbnail(wsi_path=input_path, target_mpp=target_mpp)
    if thumbnail is None:
        return None, None

    # If thumbnail has 4 channels (RGBA), drop the alpha channel
    if thumbnail.shape[2] == 4:
        thumbnail = thumbnail[:, :, :3]

    # Measure tissue detection time
    start_time = time.time()
    detector.detect_tissue(thumbnail)
    detection_time = time.time() - start_time

    # Save the mask
    file_base = os.path.splitext(os.path.basename(input_path))[0]
    mask_output_path = os.path.join(mask_output_dir, f"{file_base}.png")
    cv2.imwrite(mask_output_path, detector.mask)

    return mask_output_path, detection_time

if __name__ == "__main__":
    wsi_dir = "/home/bogdan/indonezia/data/GRAND-QC/data/wsi"
    out_dir = "/home/bogdan/indonezia/data/GRAND-QC/data/inference/kmeans_algorithm"

    for cohort_name in tqdm(os.listdir(wsi_dir), desc="Processing cohorts"):
        input_dir = os.path.join(wsi_dir, cohort_name)
        out_dir_cohort = os.path.join(out_dir, cohort_name)

        mask_output_dir = os.path.join(out_dir_cohort, "0_255")
        if not os.path.exists(mask_output_dir):
            os.makedirs(mask_output_dir)

        detector = KmeansTissueDetector(n_clusters=2)
        processing_times = []

        for file_name in tqdm(os.listdir(input_dir), desc="Processing WSIs"):
            if file_name.endswith(('.svs', '.tiff', '.tif')):
                input_path = os.path.join(input_dir, file_name)
                try:
                    mask_path, detection_time = process_wsi(input_path, mask_output_dir, detector, target_mpp=10.0)
                    if mask_path:
                        processing_times.append((file_name, detection_time))
                        print(f"Tissue detection for {file_name} took {detection_time:.2f} seconds. Mask saved at {mask_path}.")
                    else:
                        print(f"Skipped {file_name}.")
                except Exception as e:
                    print(f"Error processing {file_name}: {e}")
        csv_output_path = os.path.join(mask_output_dir, "processing_times.csv")
        with open(csv_output_path, mode='w', newline='') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(["Filename", "Processing Time (seconds)"])
            writer.writerows(processing_times)

        print(f"Processing times saved to {csv_output_path}")