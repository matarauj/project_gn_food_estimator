# =============================================================================
# pipeline/stage1_capture.py
# =============================================================================
# Stage I — image ingestion, preprocessing, and ArUco validation.
#
# Accepts images from two sources:
#   - File upload  : bytes or file-like object from st.file_uploader
#   - Camera       : bytes from st.camera_input
#
# Returns a StageOneResult containing:
#   - The preprocessed RGB image
#   - The original full-resolution RGB image (for display)
#   - A CalibrationState object (calibrated or approximate)
# 
# ArUco detection is now a soft failure: if the marker is not found the
# pipeline continues in "approximate mode" rather than stopping.
# The CalibrationState object signals this to all downstream stages.
# =============================================================================

from dataclasses import dataclass
import numpy as np
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from cv_tasks.preprocessor import preprocess, load_image_rgb, to_bgr
from cv_tasks.aruco import CalibrationState, build_calibration_state
import config as cfg


@dataclass
class StageOneResult:
    image_rgb:     np.ndarray       # preprocessed RGB image (longest edge = IMAGE_SIZE)
    image_raw_rgb: np.ndarray       # original RGB image before resize (for display)
    calibration:   CalibrationState # scale / homography state


def run(source) -> StageOneResult:
    """
    Execute Stage I.

    Parameters
    ----------
    source : bytes | str | Path | np.ndarray
        Image data from Streamlit file_uploader or camera_input.

    Returns
    -------
    StageOneResult
        calibration.mode == "calibrated"  → ArUco found, full accuracy
        calibration.mode == "approximate" → no marker, fallback scale used

    Raises
    ------
    ValueError
        If the image cannot be decoded.
    """
    # Load at full resolution (ArUco detection needs the raw image)
    image_raw = load_image_rgb(source)

    # ArUco detection on the raw image
    image_bgr  = to_bgr(image_raw)
    image_scale = cfg.IMAGE_SIZE / max(image_raw.shape[:2])
    calibration = build_calibration_state(image_bgr, image_scale)

    # Preprocess (CLAHE + resize) for model inference
    image_preprocessed = preprocess(
        image_raw,
        apply_hist_eq = cfg.HIST_EQ_ENABLED,
        resize        = True,
    )

    return StageOneResult(
        image_rgb     = image_preprocessed,
        image_raw_rgb = image_raw,
        calibration   = calibration,
    )