#!/usr/bin/env python3
"""
Script Summary:
---------------
Compute aggregated evaluation metrics (mean IoU, mean Dice coefficient, average inference time)
for each tissue detection method across all cohorts and images from a result JSON.
Saves a summary CSV to the specified output directory.

Example:
    python summarize_eval.py --result-json /home/user/eval.json --output-dir ./results_summary
"""

import os
import sys
import json
import logging
import argparse
from typing import Any, Dict, List
from collections import defaultdict

import pandas as pd
from tqdm import tqdm

# ----------------------------------------------------------------------------
# Configuration constants (modify defaults as needed)
# ----------------------------------------------------------------------------
CONFIG: Dict[str, Any] = {
    "DEFAULT_RESULT_JSON": "/home/bogdan/indonezia/data/GRAND-QC/data/eval/eval.json",
    "DEFAULT_OUTPUT_DIR": "/home/bogdan/indonezia/data/GRAND-QC/paper-tissue-detection/eval_algorithms"
}

def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Summarize evaluation JSON into CSV of metrics.")
    parser.add_argument(
        "--result-json",
        type=str,
        default=CONFIG["DEFAULT_RESULT_JSON"],
        help="Path to evaluation JSON file."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=CONFIG["DEFAULT_OUTPUT_DIR"],
        help="Directory to save the summary CSV."
    )
    return parser.parse_args()

def load_results(json_path: str) -> Dict[str, Any]:
    """
    Load the evaluation results JSON.

    Raises FileNotFoundError or json.JSONDecodeError on failure.
    """
    if not os.path.isfile(json_path):
        logging.error(f"Result JSON not found: {json_path}")
        raise FileNotFoundError(f"Result JSON not found: {json_path}")
    with open(json_path, "r") as f:
        return json.load(f)

def aggregate_metrics(
    data: Dict[str, Any]
) -> Dict[str, Dict[str, List[float]]]:
    """
    Aggregate IoU, Dice, and time metrics for each method
    across all cohorts and images.
    """
    metrics: Dict[str, Dict[str, List[float]]] = defaultdict(
        lambda: {"iou": [], "dice": [], "time": []}
    )
    for cohort, images in tqdm(data.items(), desc="Aggregating cohorts"):
        for img_name, entry in tqdm(images.items(), desc=f"{cohort}", leave=False):
            results_list = entry.get("tissue_detector_algorithm_results", [])
            for method_result in results_list:
                for method, vals in method_result.items():
                    try:
                        metrics[method]["iou"].append(float(vals.get("iou", 0.0)))
                        metrics[method]["dice"].append(float(vals.get("dice", 0.0)))
                        metrics[method]["time"].append(float(vals.get("time", 0.0)))
                    except Exception as e:
                        logging.warning(f"Failed to parse metrics for {method} on {img_name}: {e}")
    return metrics

def compute_statistics(
    metrics: Dict[str, Dict[str, List[float]]]
) -> List[Dict[str, Any]]:
    """
    Compute mean IoU, mean Dice, and average time for each method.
    Returns a list of dicts with keys: Method, mIoU, mDCC, AverageTime.
    """
    stats: List[Dict[str, Any]] = []
    for method, vals in metrics.items():
        iou_list = vals["iou"]
        dice_list = vals["dice"]
        time_list = vals["time"]
        count = len(iou_list)
        if count == 0:
            logging.warning(f"No entries found for method: {method}")
            continue
        mean_iou = sum(iou_list) / count
        mean_dice = sum(dice_list) / count
        avg_time = sum(time_list) / count
        stats.append({
            "Method": method,
            "mIoU": mean_iou,
            "mDCC": mean_dice,
            "AverageTime": avg_time
        })
    return stats

def save_csv(
    stats: List[Dict[str, Any]],
    output_dir: str
) -> None:
    """
    Save the statistics to a CSV file in the output directory.
    """
    os.makedirs(output_dir, exist_ok=True)
    df = pd.DataFrame(stats)
    csv_path = os.path.join(output_dir, "summary_metrics.csv")
    df.to_csv(csv_path, index=False)
    logging.info(f"Saved summary CSV to {csv_path}")

def main() -> None:
    """
    Main entry point.
    """
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )
    try:
        data = load_results(args.result_json)
        metrics = aggregate_metrics(data)
        stats = compute_statistics(metrics)
        save_csv(stats, args.output_dir)
    except Exception as e:
        logging.error(f"Processing failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
