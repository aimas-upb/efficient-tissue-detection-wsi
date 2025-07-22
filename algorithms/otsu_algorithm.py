import os
import csv
import time
import logging
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm
from tiatoolbox.tools.tissuemask import OtsuTissueMasker

from get_thumbnail import generate_thumbnail

# Configuration constants
CONFIG: Dict[str, Any] = {
    "cohorts_dir": "/home/bogdan/indonezia/data/GRAND-QC/data/wsi",
    "inference_dir_out": "/home/bogdan/indonezia/data/GRAND-QC/to_delete/inference",
    "target_mpp": 10.0,
    "csv_file_name": "detection_times.csv"
}

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)


class OtsuTissueDetector:
    """
    Class for performing Otsu-based tissue detection on WSI thumbnails.

    Methods:
        detect_tissue(thumbnail): Fit and transform a thumbnail to a binary tissue mask.
    """

    def __init__(self) -> None:
        """
        Initialize the Otsu tissue masker.
        """
        self.masker = OtsuTissueMasker()

    def detect_tissue(self, thumbnail: np.ndarray) -> np.ndarray:
        """
        Generate a binary tissue mask from a thumbnail image using Otsu's method.

        This first fits the Otsu model on the provided thumbnail, then transforms it.

        Args:
            thumbnail (np.ndarray): RGB thumbnail image of shape (H, W, 3) or grayscale (H, W).

        Returns:
            np.ndarray: Binary mask (uint8) of shape (H, W) with tissue=255 and background=0.

        Raises:
            ValueError: If thumbnail is empty or invalid format.
        """
        if thumbnail is None or thumbnail.size == 0:
            logging.error("Empty thumbnail provided to detect_tissue.")
            raise ValueError("Invalid thumbnail image.")

        # Convert grayscale to RGB if needed
        if thumbnail.ndim == 2:
            thumbnail = cv2.cvtColor(thumbnail, cv2.COLOR_GRAY2RGB)

        # Fit the masker on the thumbnail list
        self.masker.fit([thumbnail])
        # Transform to binary mask
        binary_mask: np.ndarray = self.masker.transform([thumbnail])[0]

        # Convert boolean mask to uint8 (0/255)
        mask_uint8: np.ndarray = (binary_mask > 0).astype(np.uint8) * 255
        return mask_uint8


def process_wsi(
    input_path: str,
    mask_output_dir: str,
    detector: OtsuTissueDetector,
    target_mpp: float = CONFIG["target_mpp"]
) -> Tuple[Optional[str], Optional[float]]:
    """
    Open a WSI, generate a thumbnail, run tissue detection, and save the mask.

    Args:
        input_path (str): Path to the WSI file.
        mask_output_dir (str): Directory to save resulting mask.
        detector (OtsuTissueDetector): Initialized detector instance.
        target_mpp (float): Target microns-per-pixel for thumbnail.

    Returns:
        Tuple[Optional[str], Optional[float]]: (mask_path, detection_time) if successful; (None, None) otherwise.
    """
    # Generate thumbnail
    thumbnail: Optional[np.ndarray] = generate_thumbnail(
        wsi_path=input_path,
        target_mpp=target_mpp
    )
    if thumbnail is None:
        logging.warning(f"Thumbnail generation failed for {input_path}.")
        return None, None

    # Perform detection
    start_time: float = time.time()
    try:
        mask: np.ndarray = detector.detect_tissue(thumbnail)
        detection_time: float = time.time() - start_time
    except Exception as e:
        logging.error(f"Error during tissue detection for {input_path}: {e}")
        return None, None

    # Save mask as PNG
    file_base: str = os.path.splitext(os.path.basename(input_path))[0]
    mask_file: str = f"{file_base}.png"
    mask_path: str = os.path.join(mask_output_dir, mask_file)
    try:
        cv2.imwrite(mask_path, mask)
        logging.info(
            f"Saved mask for {input_path} to {mask_path} "
            f"(detection time: {detection_time:.3f}s)"
        )
    except Exception as e:
        logging.error(f"Failed to save mask for {input_path}: {e}")
        return None, None

    return mask_path, detection_time


def main() -> None:
    """
    Iterate over cohorts of WSIs and apply Otsu tissue detection.
    Saves binary masks and timing information to CSV.
    """
    cohorts_dir: str = CONFIG["cohorts_dir"]
    inference_dir: str = os.path.join(CONFIG["inference_dir_out"], "otsu")
    os.makedirs(inference_dir, exist_ok=True)

    detector = OtsuTissueDetector()

    for cohort in tqdm(os.listdir(cohorts_dir), desc="Cohorts"):
        cohort_in: str = os.path.join(cohorts_dir, cohort)
        cohort_out: str = os.path.join(inference_dir, cohort)
        mask_dir: str = os.path.join(cohort_out, "0_255")
        os.makedirs(mask_dir, exist_ok=True)

        processing_records: List[Tuple[str, float]] = []

        try:
            wsi_files: List[str] = [
                f for f in os.listdir(cohort_in)
                if f.lower().endswith(('.svs', '.tiff', '.tif'))
            ]
        except Exception as e:
            logging.error(f"Failed to list WSIs in {cohort_in}: {e}")
            continue

        for wsi_file in tqdm(wsi_files, desc=f"Processing {cohort}", unit="wsi"):
            input_path = os.path.join(cohort_in, wsi_file)
            mask_path, det_time = process_wsi(
                input_path=input_path,
                mask_output_dir=mask_dir,
                detector=detector
            )
            if mask_path and det_time is not None:
                processing_records.append((wsi_file, det_time))

        # Write CSV of processing times
        csv_path: str = os.path.join(cohort_out, CONFIG["csv_file_name"])
        try:
            with open(csv_path, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(["Filename", "DetectionTime_seconds"])
                writer.writerows(processing_records)
            logging.info(f"Wrote timing CSV to {csv_path}")
        except Exception as e:
            logging.error(f"Failed to write CSV at {csv_path}: {e}")


if __name__ == "__main__":
    main()
