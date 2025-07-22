#!/usr/bin/env python3
"""
Grand QC Tissue Detector
------------------------
Implements a standardized TissueDetector interface for the GrandQC deep-learning model.

- `detect_tissue(thumbnail: np.ndarray) -> np.ndarray`: runs patch-based inference on the provided thumbnail
  and returns a uint8 mask (0/255).
- `process_wsi(wsi_path: str, target_mpp: float) -> Tuple[np.ndarray, float]`: loads the WSI thumbnail via generate_thumbnail,
  then calls `detect_tissue` on that thumbnail and times the operation.

Configuration constants defined in the global CONFIG dictionary, preserving original script logic and variable names.
"""
import os
import time
import logging
from typing import Tuple, Optional, Dict, Any

import cv2
import numpy as np
import torch
from PIL import Image
import segmentation_models_pytorch as smp
from .wsi_tis_detect_helper_fx import get_preprocessing, make_class_map

# ----------------------------------------------------------------------------
# Configuration constants (modify as needed)
# ----------------------------------------------------------------------------
CONFIG: Dict[str, Any] = {
    "DEVICE": "cuda",
    "MODEL_TD_DIR": "/home/bogdan/indonezia/data/GRAND-QC/paper-tissue-detection/algorithms/gradqc_algorithm/models/td",
    "MODEL_TD_NAME": "Tissue_Detection_MPP10.pth",
    "MPP_MODEL_TD": 10.0,
    "M_P_S_MODEL_TD": 512,
    "ENCODER_MODEL_TD": "timm-efficientnet-b0",
    "ENCODER_MODEL_TD_WEIGHTS": "imagenet",
    "JPEG_QUALITY": 80,
    # visualization overlays not used here
    "OVER_IMAGE": 0.7,
    "OVER_MASK": 0.3,
    "COLORS": [[50, 50, 250], [128, 128, 128]]
}

# Logging setup
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class GrandQCTissueDetector:
    """
    TissueDetector wrapper for the GrandQC DL model, using global CONFIG.
    """
    def __init__(self, device_str: str = "cuda:0") -> None:

        if device_str.startswith("cuda") and torch.cuda.is_available():
           self.device = torch.device(device_str)
        else:
           self.device = torch.device("cpu")

        self._load_model()

    def _load_model(self) -> None:
        self.preprocessing_fn = smp.encoders.get_preprocessing_fn(
            CONFIG["ENCODER_MODEL_TD"], CONFIG["ENCODER_MODEL_TD_WEIGHTS"]
        )
        self.model = smp.UnetPlusPlus(
            encoder_name=CONFIG["ENCODER_MODEL_TD"],
            encoder_weights=CONFIG["ENCODER_MODEL_TD_WEIGHTS"],
            classes=2,
            activation=None,
        )
        model_path = os.path.join(CONFIG["MODEL_TD_DIR"], CONFIG["MODEL_TD_NAME"])
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()

    def detect_tissue(self, thumbnail: np.ndarray) -> np.ndarray:
        """
        Run patch-based inference on the thumbnail and return a binary mask (0/255).
        Mirrors variable names: image, image_work, temp_image, end_image, and builds a color map as in original script.
        """
        # JPEG compression
        _, jpg = cv2.imencode(
            ".jpg", thumbnail,
            [int(cv2.IMWRITE_JPEG_QUALITY), CONFIG["JPEG_QUALITY"]]
        )
        image = Image.fromarray(cv2.imdecode(jpg, cv2.IMREAD_COLOR))

        width, height = image.size
        wi_n = width // CONFIG["M_P_S_MODEL_TD"]
        he_n = height // CONFIG["M_P_S_MODEL_TD"]
        overhang_wi = width - wi_n * CONFIG["M_P_S_MODEL_TD"]
        overhang_he = height - he_n * CONFIG["M_P_S_MODEL_TD"]
        p_s = CONFIG["M_P_S_MODEL_TD"]

        end_image = None
        end_image_class_map = None
        for h in range(he_n + 1):
            for w in range(wi_n + 1):
                if w != wi_n and h != he_n:
                    image_work = image.crop((w * p_s, h * p_s, (w + 1) * p_s, (h + 1) * p_s))
                elif w == wi_n and h != he_n:
                    image_work = image.crop((width - p_s, h * p_s, width, (h + 1) * p_s))
                elif w != wi_n and h == he_n:
                    image_work = image.crop((w * p_s, height - p_s, (w + 1) * p_s, height))
                else:
                    image_work = image.crop((width - p_s, height - p_s, width, height))

                image_pre = get_preprocessing(image_work, self.preprocessing_fn)
                x_tensor = torch.from_numpy(image_pre).to(self.device).unsqueeze(0)
                predictions = self.model.predict(x_tensor)
                predictions = predictions.squeeze().cpu().numpy()
                mask = np.argmax(predictions, axis=0).astype(np.uint8)
                class_mask = make_class_map(mask, CONFIG["COLORS"])

                if w == 0:
                    temp_image = mask
                    temp_image_class_map = class_mask
                elif w == wi_n:
                    mask_crop = mask[:, p_s - overhang_wi : p_s]
                    class_crop = class_mask[:, p_s - overhang_wi : p_s, :]
                    temp_image = np.concatenate((temp_image, mask_crop), axis=1)
                    temp_image_class_map = np.concatenate((temp_image_class_map, class_crop), axis=1)
                else:
                    temp_image = np.concatenate((temp_image, mask), axis=1)
                    temp_image_class_map = np.concatenate((temp_image_class_map, class_mask), axis=1)

            if h == 0:
                end_image = temp_image
                end_image_class_map = temp_image_class_map
            elif h == he_n:
                crop_row = temp_image[p_s - overhang_he : p_s, :]
                crop_class_row = temp_image_class_map[p_s - overhang_he : p_s, :, :]
                end_image = np.concatenate((end_image, crop_row), axis=0)
                end_image_class_map = np.concatenate((end_image_class_map, crop_class_row), axis=0)
            else:
                end_image = np.concatenate((end_image, temp_image), axis=0)
                end_image_class_map = np.concatenate((end_image_class_map, temp_image_class_map), axis=0)

                # Return binary 0/255 mask (ignore class_map for this API)
        return (end_image == 0).astype(np.uint8) * 255
