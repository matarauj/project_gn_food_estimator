# =============================================================================
# cv_tasks/preprocessor.py
# =============================================================================
# Image preprocessing applied in Stage I before any model inference.
# Keeps preprocessing deterministic and centralised so training and inference
# apply identical operations.
# =============================================================================

import cv2
import numpy as np
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))
import config as cfg


def load_image_rgb(source) -> np.ndarray:
    """
    Load an image from a file path, bytes buffer, or numpy array and return
    it as an RGB uint8 array.

    Accepts:
        - str / Path  : file path
        - bytes       : raw image bytes (e.g. from st.camera_input or upload)
        - np.ndarray  : already-decoded image (BGR or RGB, uint8)
    """
    # Read image path
    if isinstance(source, (str, Path)):
        bgr = cv2.imread(str(source))
        if bgr is None:
            raise FileNotFoundError(f"Could not read image: {source}")
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    # Decode image bytes
    if isinstance(source, (bytes, bytearray)):
        arr = np.frombuffer(source, dtype = np.uint8)
        bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError("Could not decode image bytes.")
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    
    # Array
    if isinstance(source, np.ndarray):
        # Grayscale
        if source.ndim == 2:
            return cv2.cvtColor(source, cv2.COLOR_GRAY2RGB)
        # 4 channels (RGB + alpha)
        if source.shape[2] == 4:
            return cv2.cvtColor(source, cv2.COLOR_RGBA2RGB)
        return source.copy()

    raise TypeError(f"Unsupported image source type: {type(source)}")


def histogram_equalisation(image_rgb: np.ndarray) -> np.ndarray:
    """
    Apply CLAHE (Contrast Limited Adaptive Histogram Equalisation) to the
    luminance channel of an RGB image.

    CLAHE is preferred over global histogram equalisation because it avoids
    over-amplifying noise in already-uniform regions (e.g. empty containers).

    Returns
        RGB uint8 array of the same shape.
    """
    lab   = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit = 2.0, tileGridSize = (8, 8))
    l_eq  = clahe.apply(l)
    lab_eq = cv2.merge([l_eq, a, b])
    return cv2.cvtColor(lab_eq, cv2.COLOR_LAB2RGB)


def resize_longest_edge(
        image_rgb: np.ndarray,
        size: int = cfg.IMAGE_SIZE) -> np.ndarray:
    """
    Resize so the longest edge equals `size`, preserving aspect ratio.
    Uses INTER_AREA for downscaling (sharpest result for photos).
    """
    h, w  = image_rgb.shape[:2]
    scale = size / max(h, w)
    if abs(scale - 1.0) < 0.01:
        return image_rgb
    
    new_w = int(w * scale)
    new_h = int(h * scale)
    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    return cv2.resize(image_rgb, (new_w, new_h), interpolation = interp)


def preprocess(image_rgb: np.ndarray,
               apply_hist_eq: bool = cfg.HIST_EQ_ENABLED,
               resize: bool = True) -> np.ndarray:
    """
    Full preprocessing pipeline for a single image.

    Steps:
        1. Load → RGB uint8
        2. Histogram equalisation (CLAHE) if enabled
        3. Resize longest edge to IMAGE_SIZE

    Parameters
    ----------
    source : str | Path | bytes | np.ndarray
    apply_hist_eq : bool
    resize : bool

    Returns
    -------
    np.ndarray  RGB uint8, longest edge = IMAGE_SIZE
    """
    image = image_rgb.copy()
    
    if apply_hist_eq:
        image = histogram_equalisation(image)

    if resize:
        image = resize_longest_edge(image)
    return image


def to_bgr(image_rgb: np.ndarray) -> np.ndarray:
    """
    Convert RGB → BGR for any cv2 functions that need it.
    """
    return cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)