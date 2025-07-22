#!/usr/bin/env python3
"""
Script Summary:
---------------
Evaluate multiple tissue detection algorithms on a dataset index JSON.
For each configured detector, runs the algorithm on thumbnails (timing only detection),
computes IoU and Dice against ground truth masks, saves predicted masks, and
incrementally updates a single JSON report `eval.json` in OUTPUT_DIR.
"""
import os
import json
import time
import logging
from typing import Any, Dict, Optional

import cv2
import numpy as np
from tqdm import tqdm

from algorithms.otsu_algorithm import OtsuTissueDetector
from algorithms.kmeans_algorithm import KmeansTissueDetector
from algorithms.combined_detector_algorithm import DoublePass
from algorithms.gradqc_algorithm.tissue_detector_grandqc import GrandQCTissueDetector
from algorithms.get_thumbnail import generate_thumbnail

# ----------------------------------------------------------------------------
# Configuration constants (modify as needed)
# ----------------------------------------------------------------------------
CONFIG: Dict[str, Any] = {
    "DATASET_INDEX_JSON": "/home/bogdan/indonezia/data/GRAND-QC/data/dataset/tissue_detection_mpp_10/dataset_index.json",
    "OUTPUT_DIR": "/home/bogdan/indonezia/data/GRAND-QC/data/eval",
    "TARGET_MPP": 10.0,
    "IMAGE_EXTENSION": ".png"
}

# Map method names to detector constructors
DETECTOR_MAP: Dict[str, Any] = {
    "otsu": lambda: OtsuTissueDetector(),
    "kmeans": lambda: KmeansTissueDetector(n_clusters=2),
    "double_pass": lambda: DoublePass(),
    "grandqc_on_gpu": lambda: GrandQCTissueDetector("cuda:0"),
    "grandqc_on_cpu": lambda: GrandQCTissueDetector("cpu"),
    
}

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def compute_metrics(gt_mask: np.ndarray, pred_mask: np.ndarray) -> Dict[str, float]:
    """
    Compute IoU and Dice Score for binary masks with values {0,255}.
    """
    if gt_mask.shape != pred_mask.shape:
        pred_mask = cv2.resize(
            pred_mask,
            (gt_mask.shape[1], gt_mask.shape[0]),
            interpolation=cv2.INTER_NEAREST
        )
    gt_bin = (gt_mask == 255).astype(np.uint8)
    pred_bin = (pred_mask == 255).astype(np.uint8)
    intersection = int(np.sum(gt_bin * pred_bin))
    union = int(np.sum((gt_bin + pred_bin) > 0))
    iou = 1.0 if union == 0 else intersection / union
    sum_masks = int(np.sum(gt_bin) + np.sum(pred_bin))
    dice = 1.0 if sum_masks == 0 else 2 * intersection / sum_masks
    return {"iou": float(iou), "dice": float(dice)}


def process_wsi(
    img_name: str,
    meta: Dict[str, Any],
    detector: Any,
    output_subdir: str,
    target_mpp: float
) -> Optional[Dict[str, Any]]:
    """
    Process a single WSI entry: generate thumbnail, detect tissue, save mask, compute metrics.
    Returns a result dict for one method.
    """
    wsi_path = meta.get("wsi_path")
    gt_mask_path = meta.get("gt_mask")

    thumbnail = generate_thumbnail(wsi_path=wsi_path, target_mpp=target_mpp)
    if thumbnail is None:
        logger.warning(f"Thumbnail generation failed for {img_name}")
        return None

    try:
        start = time.time()
        pred_mask = detector.detect_tissue(thumbnail)
        elapsed = time.time() - start
    except Exception as e:
        logger.error(f"Detection failed for {img_name}: {e}")
        return None

    mask_file = os.path.splitext(img_name)[0] + CONFIG["IMAGE_EXTENSION"]
    pred_mask_path = os.path.join(output_subdir, mask_file)
    cv2.imwrite(pred_mask_path, pred_mask)

    gt_mask = cv2.imread(gt_mask_path, cv2.IMREAD_GRAYSCALE)
    if gt_mask is None:
        logger.error(f"Failed loading ground truth for {img_name}")
        return None

    metrics = compute_metrics(gt_mask, pred_mask)

    return {
        "pred_mask_path": pred_mask_path,
        "iou": metrics["iou"],
        "dice": metrics["dice"],
        "time": elapsed
    }


def main() -> None:
    # Load dataset index
    with open(CONFIG["DATASET_INDEX_JSON"]) as f:
        dataset_index: Dict[str, Dict[str, Any]] = json.load(f)

    # Prepare single results file at root of OUTPUT_DIR
    os.makedirs(CONFIG["OUTPUT_DIR"], exist_ok=True)
    results_file = os.path.join(CONFIG["OUTPUT_DIR"], "eval.json")

    # Load existing or init new structure
    if os.path.isfile(results_file):
        with open(results_file) as rf:
            final_results: Dict[str, Dict[str, Any]] = json.load(rf)
    else:
        final_results = {}

    # Ensure top-level cohorts exist
    for cohort in dataset_index:
        final_results.setdefault(cohort, {})

    # Evaluate each detector in DETECTOR_MAP
    for method_name, ctor in DETECTOR_MAP.items():
        logger.info(f"Evaluating method: {method_name}")
        detector = ctor()

        for cohort, entries in dataset_index.items():
            mask_out_dir = os.path.join(CONFIG["OUTPUT_DIR"], method_name, cohort, "0_255")
            os.makedirs(mask_out_dir, exist_ok=True)

            for img_name, meta in tqdm(entries.items(), desc=f"{method_name}: {cohort}"):
                final_results[cohort].setdefault(
                    img_name,
                    {**meta, "tissue_detector_algorithm_results": []}
                )
                existing = final_results[cohort][img_name]["tissue_detector_algorithm_results"]

                # Skip if this method already evaluated
                if any(method_name in d for d in existing):
                    continue

                result = process_wsi(
                    img_name=img_name,
                    meta=meta,
                    detector=detector,
                    output_subdir=mask_out_dir,
                    target_mpp=CONFIG["TARGET_MPP"]
                )
                if result:
                    final_results[cohort][img_name]["tissue_detector_algorithm_results"].append({
                        method_name: result
                    })
                    # Save incrementally
                    with open(results_file, 'w') as wf:
                        json.dump(final_results, wf, indent=4)

    logger.info(f"Saved consolidated results to {results_file}")


if __name__ == "__main__":
    main()
