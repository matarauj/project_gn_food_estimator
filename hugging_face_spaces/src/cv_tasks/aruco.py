# =============================================================================
# cv_tasks/aruco.py
# =============================================================================
# ArUco marker detection and pixel-to-mm scale estimation.
# Used by Stage I (validation) and Stage III (scale for homography).
# =============================================================================

import cv2
import numpy as np
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))
import config as cfg


class ArUcoNotFoundError(Exception):
    """
    Raised when the expected ArUco marker is not detected in the image.
    """
    pass



def _get_aruco_dict():
    """
    Return the cv2.aruco dictionary specified in config.
    """
    dict_map = {
        "DICT_4X4_50":   cv2.aruco.DICT_4X4_50,
        "DICT_4X4_100":  cv2.aruco.DICT_4X4_100,
        "DICT_6X6_250":  cv2.aruco.DICT_6X6_250
    }
    dict_id = dict_map.get(cfg.ARUCO_DICT_ID)
    if dict_id is None:
        raise ValueError(f"Unknown ArUco dict: {cfg.ARUCO_DICT_ID}")
    return cv2.aruco.getPredefinedDictionary(dict_id)


def detect_aruco(image_bgr: np.ndarray) -> dict | None:
    """
    Detect the configured ArUco marker in an BGR image.
    It tries to detect just one marker.

    Parameters
    ----------
    image_bgr : np.ndarray
        Full-scene image in BGR format (as returned by cv2.imread or
        cv2.cvtColor from RGB).

    Returns
    -------
    dict with keys:
        corners   : np.ndarray shape (4, 2) — marker corner pixel coords
                    in order: top-left, top-right, bottom-right, bottom-left
        center    : np.ndarray shape (2,)  — marker centre pixel coords
        side_px   : float — mean side length in pixels
        mm_per_px : float — scale factor (mm per pixel)
    Returns None if the marker is not found.
    """
    aruco_dict   = _get_aruco_dict()
    params       = cv2.aruco.DetectorParameters()
    detector     = cv2.aruco.ArucoDetector(aruco_dict, params)
    corners_list, ids, _ = detector.detectMarkers(image_bgr)

    if ids is None:
        return None

    # Find our specific marker ID
    ids_flat = ids.flatten()
    matches  = np.where(ids_flat == cfg.ARUCO_MARKER_ID)[0]
    if len(matches) == 0:
        return None

    # Take the first match (there should only be one)
    corners = corners_list[matches[0]][0]   # shape (4, 2)

    # Compute mean side length in pixels
    sides = [
        np.linalg.norm(corners[1] - corners[0]),
        np.linalg.norm(corners[2] - corners[1]),
        np.linalg.norm(corners[3] - corners[2]),
        np.linalg.norm(corners[0] - corners[3])
    ]
    side_px   = float(np.mean(sides))
    # Ratio mm per pixel
    mm_per_px = cfg.ARUCO_MARKER_MM / side_px
    center    = corners.mean(axis = 0)

    return {
        "corners":   corners,
        "center":    center,
        "side_px":   side_px,
        "mm_per_px": mm_per_px
    }


def require_aruco(image_bgr: np.ndarray) -> dict:
    """
    Same as detect_aruco but raises ArUcoNotFoundError if not found.
    """
    result = detect_aruco(image_bgr)
    if result is None:
        raise ArUcoNotFoundError(
            f"ArUco marker (ID {cfg.ARUCO_MARKER_ID}, dict {cfg.ARUCO_DICT_ID}) "
            "not detected. Please ensure the marker is visible and retake the photo."
        )
    return result


def draw_aruco_overlay(image_bgr: np.ndarray, aruco_result: dict) -> np.ndarray:
    """
    Draw the detected ArUco marker corners and scale annotation on a copy
    of the image. Useful for the Stage I debug panel in the app.

    Returns a BGR image with the overlay drawn.
    """
    overlay = image_bgr.copy()
    corners = aruco_result["corners"].astype(int)

    # Draw marker outline
    cv2.polylines(
        overlay,
        [corners],
        isClosed = True,
        color = (0, 220, 130),
        thickness = 2
    )

    # Draw corner dots
    for pt in corners:
        cv2.circle(overlay, tuple(pt), 5, (0, 220, 130), -1)

    # Scale annotation
    cx, cy   = aruco_result["center"].astype(int)
    scale_txt = f"{aruco_result['mm_per_px']:.3f} mm/px"
    cv2.putText(
        overlay,
        scale_txt,
        (cx - 60, cy - 15),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 220, 130),
        2
    )

    return overlay