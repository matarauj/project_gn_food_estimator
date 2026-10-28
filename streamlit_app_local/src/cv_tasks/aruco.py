# =============================================================================
# cv_tasks/aruco.py
# =============================================================================
# ArUco marker detection, pixel-to-mm scale estimation, and CalibrationState
# Used by Stage I (validation + CalibrationState creation) and
# Stage III (scale for homography / approximate measurements).
# =============================================================================

from __future__ import annotations
from dataclasses import dataclass
from typing import Literal

import cv2
import numpy as np
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))
import config as cfg


# =============================================================================
# CalibrationState
# =============================================================================

@dataclass
class CalibrationState:
    """
    Carries ArUco-derived (or fallback) scale information through the pipeline.

    Attributes
    ----------
    mode : "calibrated" | "approximate"
        "calibrated"  — ArUco marker was found; all fields are populated.
        "approximate" — No marker found; homography is None, mm_per_px is
                        a configured fallback constant, measurement_reliable
                        is False.
    aruco_detected : bool
    mm_per_px : float
        Scale factor.
        Measured from the marker in calibrated mode;
        ARUCO_FALLBACK_MM_PER_PX in approximate mode.
    homography : np.ndarray | None
        3x3 perspective-correction matrix.
        None in approximate mode.
    marker_id : int | None
        ID of the detected marker.
        None in approximate mode.
    corners : np.ndarray | None
        Shape (4, 2) marker corners in the *rescaled* image.
        None in approximate mode.
    center : np.ndarray | None
        Shape (2,) marker centre in the *rescaled* image.
        None in approximate mode.
    side_px : float | None
        Mean side length of the detected marker in pixels (rescaled image).
    measurement_reliable : bool
        True only in calibrated mode.
        Downstream stages and the UI use this to flag that volume/mass estimates are approximate.
    """
    mode:                Literal["calibrated", "approximate"]
    aruco_detected:      bool
    mm_per_px:           float
    homography:          np.ndarray | None
    marker_id:           int | None
    corners:             np.ndarray | None
    center:              np.ndarray | None
    side_px:             float | None
    measurement_reliable: bool

    # ------------------------------------------------------------------
    # Convenience constructors
    # ------------------------------------------------------------------

    @classmethod
    def calibrated(
        cls,
        mm_per_px: float,
        homography: np.ndarray,
        marker_id: int,
        corners: np.ndarray,
        center: np.ndarray,
        side_px: float,
    ) -> "CalibrationState":
        return cls(
            mode                 = "calibrated",
            aruco_detected       = True,
            mm_per_px            = mm_per_px,
            homography           = homography,
            marker_id            = marker_id,
            corners              = corners,
            center               = center,
            side_px              = side_px,
            measurement_reliable = True
        )

    @classmethod
    def approximate(cls) -> "CalibrationState":
        return cls(
            mode                 = "approximate",
            aruco_detected       = False,
            mm_per_px            = cfg.ARUCO_FALLBACK_MM_PER_PX,
            homography           = None,
            marker_id            = None,
            corners              = None,
            center               = None,
            side_px              = None,
            measurement_reliable = False
        )

    # Backwards-compat: let code that previously used aruco["mm_per_px"] etc.
    # access the same keys via dict-style lookup.
    def __getitem__(self, key: str):
        _map = {
            "mm_per_px": self.mm_per_px,
            "corners":   self.corners,
            "center":    self.center,
            "side_px":   self.side_px
        }
        if key not in _map:
            raise KeyError(key)
        return _map[key]

    def get(self, key: str, default=None):
        try:
            return self[key]
        except KeyError:
            return default


# =============================================================================
# Exceptions
# =============================================================================

class ArUcoNotFoundError(Exception):
    """
    Raised when the expected ArUco marker is not detected in the image.
    """
    pass


# =============================================================================
# Internal helpers
# =============================================================================

def _get_aruco_dict() -> cv2.aruco.Dictionary:
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


def _enhance_for_detection(image_bgr: np.ndarray) -> np.ndarray:
    """
    Apply global histogram equalisation to the grayscale channel to boost
    ArUco detection in low-contrast or uneven-lighting conditions.
    """
    gray    = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    eq      = cv2.equalizeHist(gray)
    eq_bgr  = cv2.cvtColor(eq, cv2.COLOR_GRAY2BGR)
    return eq_bgr


def _detect_raw(image_bgr: np.ndarray) -> tuple:
    """
    Run the ArucoDetector on image_bgr.
    Returns (corners_list, ids) — ids may be None.
    """
    aruco_dict = _get_aruco_dict()
    params     = cv2.aruco.DetectorParameters()
    detector   = cv2.aruco.ArucoDetector(aruco_dict, params)

    enhanced = _enhance_for_detection(image_bgr)
    corners_list, ids, _ = detector.detectMarkers(enhanced)
    return corners_list, ids


def _pick_best_marker(
    corners_list,
    ids: np.ndarray
    ) -> tuple[np.ndarray, int] | tuple[None, None]:
    """
    From all detected markers, pick the one whose ID is in ARUCO_VALID_IDS
    and that has the largest mean side length (most reliable measurement).

    Returns (corners, marker_id) or (None, None) if no valid marker found.
    """
    ids_flat = ids.flatten()
    best_corners  = None
    best_id       = None
    best_side_px  = -1.0

    for i, marker_id in enumerate(ids_flat):
        if int(marker_id) not in cfg.ARUCO_VALID_IDS:
            continue

        corners = corners_list[i][0]   # shape (4, 2)
        sides = [
            np.linalg.norm(corners[1] - corners[0]),
            np.linalg.norm(corners[2] - corners[1]),
            np.linalg.norm(corners[3] - corners[2]),
            np.linalg.norm(corners[0] - corners[3])
        ]
        side_px = float(np.mean(sides))

        if side_px > best_side_px:
            best_side_px = side_px
            best_corners = corners
            best_id      = int(marker_id)

    return best_corners, best_id


# =============================================================================
# Public API
# =============================================================================

def detect_aruco(image_bgr: np.ndarray) -> dict | None:
    """
    Detect the configured ArUco marker in an BGR image.
    Accepts any marker ID listed in cfg.ARUCO_VALID_IDS.  If multiple valid
    markers are present, the one with the largest mean side length is used.

    Parameters
    ----------
    image_bgr : np.ndarray
        Full-scene image in BGR format.

    Returns
    -------
    dict with keys:
        corners   : np.ndarray shape (4, 2)
                    marker corner pixel coords in order: top-left, top-right, bottom-right, bottom-left
        center    : np.ndarray shape (2,)
                    marker centre pixel coords
        side_px   : float — mean side length in pixels
        mm_per_px : float — scale factor
    Returns None if the marker is not found.
    """
    corners_list, ids = _detect_raw(image_bgr)

    if ids is None or len(ids) == 0:
        return None

    corners, marker_id = _pick_best_marker(corners_list, ids)
    if corners is None:
        return None

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
    center    = corners.mean(axis=0)

    return {
        "corners":   corners,
        "center":    center,
        "side_px":   side_px,
        "mm_per_px": mm_per_px,
        "marker_id": marker_id
    }


def require_aruco(image_bgr: np.ndarray) -> dict:
    """
    Same as detect_aruco but raises ArUcoNotFoundError if not found.
    Kept for compatibility with any code that needs the strict mode.
    """
    result = detect_aruco(image_bgr)
    if result is None:
        raise ArUcoNotFoundError(
            f"No valid ArUco marker (dict {cfg.ARUCO_DICT_ID}, "
            f"IDs {cfg.ARUCO_VALID_IDS}) detected. "
            "Please ensure the marker is visible and retake the photo."
        )
    return result


def build_calibration_state(
    image_bgr: np.ndarray,
    image_scale: float
    ) -> CalibrationState:
    """
    Attempt ArUco detection and return a fully populated CalibrationState.

    Detection is performed on the full-resolution BGR image.
    Corner coordinates are then scaled by `image_scale` so they align with the
    preprocessed (resized) image used by subsequent pipeline stages.
    The homography is also computed here so that Stage III only needs to
    call rectify_image().

    Parameters
    ----------
    image_bgr   : full-resolution BGR image (before resize)
    image_scale : cfg.IMAGE_SIZE / max(original_h, original_w)

    Returns
    -------
    CalibrationState — either "calibrated" or "approximate"
    """
    from cv_tasks.homography import compute_homography

    raw_result = detect_aruco(image_bgr)

    if raw_result is None:
        return CalibrationState.approximate()

    # Scale corners to match the resized inference image
    corners_scaled = raw_result["corners"] * image_scale
    center_scaled  = raw_result["center"]  * image_scale
    mm_per_px      = raw_result["mm_per_px"]   # physical ratio — scale-invariant

    homography = compute_homography(corners_scaled, mm_per_px=mm_per_px)

    return CalibrationState.calibrated(
        mm_per_px  = mm_per_px,
        homography = homography,
        marker_id  = raw_result["marker_id"],
        corners    = corners_scaled,
        center     = center_scaled,
        side_px    = raw_result["side_px"] * image_scale
    )


def draw_aruco_overlay(
        image_bgr: np.ndarray,
        calibration: CalibrationState) -> np.ndarray:
    """
    Draw the detected ArUco marker corners and scale annotation on a copy
    of the image.
    In approximate mode, draws a text banner instead of marker corners.

    Parameters
    ----------
    image_bgr   : BGR image to annotate
    calibration : CalibrationState from Stage I

    Returns a BGR image with the overlay drawn.
    """
    overlay = image_bgr.copy()
    h, w = overlay.shape[:2]

    if not calibration.aruco_detected:
        # Approximate mode banner
        cv2.putText(
            overlay,
            "ArUco NOT detected — approximate mode",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 100, 255),
            2
        )
        cv2.putText(
            overlay,
            f"fallback: {calibration.mm_per_px:.3f} mm/px",
            (10, 58),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 100, 255),
            2
        )
        return overlay

    # Calibrated mode — draw corners and scale annotation
    corners = calibration.corners.astype(int)
    cv2.polylines(overlay, [corners], isClosed=True, color=(0, 220, 130), thickness=2)
    for pt in corners:
        cv2.circle(overlay, tuple(pt), 5, (0, 220, 130), -1)

    cx, cy = calibration.center.astype(int)
    scale_txt = (
        f"ID {calibration.marker_id} | "
        f"{calibration.mm_per_px:.3f} mm/px"
    )
    cv2.putText(
        overlay,
        scale_txt,
        (cx - 80, cy - 15),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 220, 130),
        2
    )
    return overlay