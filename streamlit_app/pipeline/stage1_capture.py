# =============================================================================
# pipeline/stage1_capture.py
# =============================================================================
# Stage I — image ingestion, preprocessing, and ArUco validation.
#
# Accepts images from two sources:
#   - File upload  : bytes or file-like object from st.file_uploader
#   - Camera       : bytes from st.camera_input
#
# Returns a StageOneResult containing the preprocessed RGB image, the raw
# original, and the ArUco detection result.
# Raises ArUcoNotFoundError if the marker is absent — the app catches this
# and prompts the user to retake the photo.
# =============================================================================

from dataclasses import dataclass
import numpy as np
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from cv_tasks.preprocessor import preprocess, load_image_rgb
from cv_tasks.aruco import require_aruco, ArUcoNotFoundError
import config as cfg


@dataclass
class StageOneResult:
    image_rgb:     np.ndarray   # preprocessed RGB image (longest edge = IMAGE_SIZE)
    image_raw_rgb: np.ndarray   # original RGB image before resize (for display)
    aruco:         dict         # ArUco detection result from cv_tasks.aruco


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

    Raises
    ------
    ArUcoNotFoundError
        If the ArUco marker is not detected in the image.
    ValueError
        If the image cannot be decoded.
    """
    # Load original at full resolution for ArUco detection
    # (ArUco detection is more reliable on the unresized image)
    image_raw = load_image_rgb(source)

    # Detect ArUco on the raw image — raises if not found
    import cv2
    from cv_tasks.preprocessor import to_bgr

    image_bgr = to_bgr(image_raw)
    aruco_result = require_aruco(image_bgr)

    # Now preprocess (hist eq + resize) for model inference
    image_preprocessed = preprocess(
        image_raw,
        apply_hist_eq = cfg.HIST_EQ_ENABLED,
        resize = True
    )

    # Scale ArUco corners to match the resized image
    scale = cfg.IMAGE_SIZE / max(image_raw.shape[:2])
    aruco_scaled = dict(aruco_result)
    aruco_scaled["corners"] = aruco_result["corners"] * scale
    aruco_scaled["center"]  = aruco_result["center"]  * scale
    # mm_per_px stays the same — it is a physical ratio, not pixel-dependent

    return StageOneResult(
        image_rgb = image_preprocessed,
        image_raw_rgb = image_raw,
        aruco = aruco_scaled
    )