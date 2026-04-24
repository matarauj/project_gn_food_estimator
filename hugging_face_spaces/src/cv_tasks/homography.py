# =============================================================================
# cv_tasks/homography.py
# =============================================================================
# Perspective correction (homography) using ArUco marker corners.
# Applied in Stage III after container detection.
# =============================================================================

import cv2
import numpy as np
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))
import config as cfg


def compute_homography(aruco_corners: np.ndarray,
                       marker_mm: float = cfg.ARUCO_MARKER_MM,
                       mm_per_px: float = None) -> np.ndarray:
    """
    Compute a homography matrix that maps the observed (potentially tilted)
    ArUco marker corners to a perfect square, correcting perspective.

    The ArUco marker is the reference object: we know its real-world shape
    (a square of side marker_mm). Any deviation from a square in the image
    means the camera was not perfectly perpendicular — the homography corrects
    this.

    Parameters
    ----------
    aruco_corners : np.ndarray shape (4, 2)
        Detected marker corners in pixel coords:
        [top-left, top-right, bottom-right, bottom-left]
    marker_mm : float
        Physical side length of the printed marker.
    mm_per_px : float, optional
        If provided, the destination square is scaled to match actual pixel
        density. If None, the mean detected side length is used.

    Returns
    -------
    np.ndarray shape (3, 3) — homography matrix H
        Apply with: cv2.warpPerspective(image, H, (out_w, out_h))
    """
    # Mean side of detected marker in pixels
    sides = [
        np.linalg.norm(aruco_corners[1] - aruco_corners[0]),
        np.linalg.norm(aruco_corners[2] - aruco_corners[1]),
        np.linalg.norm(aruco_corners[3] - aruco_corners[2]),
        np.linalg.norm(aruco_corners[0] - aruco_corners[3])
    ]
    side_px = float(np.mean(sides))

    # Destination: a perfect square centred at the marker's detected centre
    cx, cy = aruco_corners.mean(axis = 0)
    half   = side_px / 2.0
    dst_corners = np.array(
        [
            [cx - half, cy - half],
            [cx + half, cy - half],
            [cx + half, cy + half],
            [cx - half, cy + half]
        ],
        dtype = np.float32
        )

    H, _ = cv2.findHomography(
        aruco_corners.astype(np.float32),
        dst_corners,
        method = 0      # exact solve — sufficient with 4 clean points
    )
    return H


def rectify_image(image_rgb: np.ndarray, H: np.ndarray) -> np.ndarray:
    """
    Apply homography H to image_rgb, producing a perspective-corrected image
    of the same dimensions.

    Parameters
    ----------
    image_rgb : np.ndarray   H x W x 3 uint8
    H         : np.ndarray   3x3 homography matrix

    Returns
    -------
    np.ndarray  H x W x 3 uint8 — rectified image
    """
    h, w = image_rgb.shape[:2]
    return cv2.warpPerspective(
        image_rgb,
        H,
        (w, h),
        flags = cv2.INTER_LINEAR,
        borderMode = cv2.BORDER_REPLICATE)


def crop_box(image_rgb: np.ndarray,
             box: list | np.ndarray,
             padding_px: int = 8) -> np.ndarray:
    """
    Crop a bounding box region from a rectified image with optional padding.

    Parameters
    ----------
    image_rgb  : np.ndarray  H x W x 3
    box        : [x_min, y_min, x_max, y_max] in pixel coords
    padding_px : int — extra pixels to add on each side

    Returns
    -------
    np.ndarray  cropped region, uint8 RGB
    """
    h, w  = image_rgb.shape[:2]
    x_min = max(0, int(box[0]) - padding_px)
    y_min = max(0, int(box[1]) - padding_px)
    x_max = min(w, int(box[2]) + padding_px)
    y_max = min(h, int(box[3]) + padding_px)
    return image_rgb[y_min:y_max, x_min:x_max].copy()


def measure_box_mm(box: list | np.ndarray,
                   mm_per_px: float) -> dict:
    """
    Convert a pixel bounding box to real-world measurements.

    Parameters
    ----------
    box        : [x_min, y_min, x_max, y_max] pixels
    mm_per_px  : scale from ArUco detection

    Returns
    -------
    dict with keys: width_mm, height_mm, area_mm2
    """
    x_min, y_min, x_max, y_max = box
    w_px = x_max - x_min
    h_px = y_max - y_min
    
    return {
        "width_mm":  w_px * mm_per_px,
        "height_mm": h_px * mm_per_px,
        "area_mm2":  w_px * h_px * mm_per_px ** 2,
    }