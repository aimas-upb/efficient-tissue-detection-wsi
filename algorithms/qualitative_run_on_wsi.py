#!/usr/bin/env python3
"""
run_all_detectors.py
====================
Runs four tissue-detection algorithms (Otsu, K-Means, Double-Pass, GrandQC)
on one WSI and writes:

  • thumbnail.png
  • otsu_mask.png, kmeans_mask.png, combined_detector_mask.png, grandqc_mask.png
  • quality_comparation_tb_sputum.png   ← Original full width, then 2 per row
"""
from __future__ import annotations

import contextlib
import logging
import os
import sys
import time
from typing import Any, Dict, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch  # type: ignore
from PIL import Image
from tqdm import tqdm

from get_thumbnail import generate_thumbnail
from otsu_algorithm import OtsuTissueDetector
from kmeans_algorithm import KmeansTissueDetector
from combined_detector_algorithm import DoublePass
from gradqc_algorithm.tissue_detector_grandqc import GrandQCTissueDetector

# --------------------------------------------------------------------------- #
# CONFIG – edit paths / parameters here
# --------------------------------------------------------------------------- #
CONFIG: Dict[str, Any] = {
    "WSI_PATH":
        "/home/bogdan/indonezia/data/positive/january_2024/wd5/"
        "Scan Slide Biogen_Part1/box6/wd5_Scan_Slide_Biogen_Part1_box6_1000613-6.svs",
    "OUTPUT_DIR":
        "/home/bogdan/indonezia/data/GRAND-QC/paper-tissue-detection/"
        "eval_algorithms/qualitative_results/tuberculosis_sputum",
    "TARGET_MPP":     10.0,      # thumbnail resolution (µm/px)
    "GRANDQC_DEVICE": "cuda:0",  # or "cpu"
    "ALPHA":          0.40,      # overlay transparency
}

# --------------------------------------------------------------------------- #
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)8s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

def save_png(img: np.ndarray, dest: str) -> None:
    if not cv2.imwrite(dest, img):
        raise IOError(f"cv2.imwrite failed for {dest}")

def overlay_rgb(rgb: np.ndarray, mask: np.ndarray,
                alpha: float = CONFIG["ALPHA"]) -> Image.Image:
    m_bool = mask.astype(bool)
    base   = Image.fromarray(rgb).convert("RGBA")
    lay    = np.zeros((*m_bool.shape, 4), np.uint8)
    lay[..., 1] = 255
    lay[..., 3] = m_bool.astype(np.uint8) * int(255 * alpha)
    return Image.alpha_composite(base, Image.fromarray(lay, "RGBA"))

def run_detector(name: str, det: Any, thumb: np.ndarray, out_dir: str
                 ) -> Tuple[np.ndarray, float]:
    t0   = time.perf_counter()
    mask = det.detect_tissue(thumb)
    sec  = time.perf_counter() - t0
    save_png(mask, os.path.join(out_dir, f"{name.lower()}_mask.png"))
    logger.info("%s finished in %.3f s", name.capitalize(), sec)
    return mask, sec

# --------------------------------------------------------------------------- #
def main() -> None:
    wsi_path, out_dir = CONFIG["WSI_PATH"], CONFIG["OUTPUT_DIR"]
    if not os.path.isfile(wsi_path):
        sys.exit(f"WSI not found: {wsi_path}")
    os.makedirs(out_dir, exist_ok=True)

    # 1) thumbnail ----------------------------------------------------------
    logger.info("Thumbnail at %.1f µm/px …", CONFIG["TARGET_MPP"])
    thumb = generate_thumbnail(wsi_path, CONFIG["TARGET_MPP"])
    if thumb is None:
        sys.exit("Thumbnail generation failed.")
    save_png(cv2.cvtColor(thumb, cv2.COLOR_RGB2BGR),
             os.path.join(out_dir, "thumbnail.png"))

    # 2) detectors ----------------------------------------------------------
    device = CONFIG["GRANDQC_DEVICE"]
    if device.startswith("cuda") and not torch.cuda.is_available():
        logger.warning("CUDA unavailable – GrandQC will use CPU")
        device = "cpu"

    detectors = {
        "otsu":              OtsuTissueDetector(),
        "kmeans":            KmeansTissueDetector(n_clusters=2),
        "combined_detector": DoublePass(),
        "grandqc":           GrandQCTissueDetector(device_str=device),
    }

    overlays: Dict[str, Image.Image] = {}
    for name, det in tqdm(detectors.items(), desc="Detectors", unit="model"):
        ctx = torch.no_grad() if isinstance(det, GrandQCTissueDetector) \
              else contextlib.nullcontext()
        with ctx:
            mask, _ = run_detector(name, det, thumb, out_dir)
            overlays[name] = overlay_rgb(thumb, mask)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # 3) qualitative strip (Original full-width, rest 2 per row) ------------
    logger.info("Saving qualitative strip …")
    titles_imgs = [
        ("Original",     Image.fromarray(thumb)),
        ("Otsu",         overlays.get("otsu")),
        ("K-Means",      overlays.get("kmeans")),
        ("Double-Pass",  overlays.get("combined_detector")),
        ("GrandQC",      overlays.get("grandqc")),
    ]
    panels = [(t, img) for t, img in titles_imgs if img is not None]

    # compute grid: first row = original spanning both cols; then 2 cols per row
    rest = panels[1:]
    n_rest = len(rest)
    n_cols = 2
    n_rest_rows = (n_rest + n_cols - 1) // n_cols
    total_rows = 1 + n_rest_rows

    fig = plt.figure(figsize=(8, 4 * total_rows))
    gs = fig.add_gridspec(total_rows, n_cols)

    # original full-width
    title0, img0 = panels[0]
    ax0 = fig.add_subplot(gs[0, :])
    ax0.imshow(img0)
    ax0.set_title(title0, fontsize=15)
    ax0.axis("off")

    # rest of the panels
    for idx, (title, img) in enumerate(rest):
        row = 1 + idx // n_cols
        col = idx % n_cols
        ax = fig.add_subplot(gs[row, col])
        ax.imshow(img)
        ax.set_title(title, fontsize=15)
        ax.axis("off")

    fig.tight_layout(pad=0.3)
    result = os.path.join(out_dir, "quality_comparation_tb_sputum.png")
    fig.savefig(result, dpi=450, bbox_inches="tight")
    plt.close(fig)
    logger.info("Figure saved → %s", result)

# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    main()
