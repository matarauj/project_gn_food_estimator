# =============================================================================
# pipeline/stage3_geometry.py
# =============================================================================
# Stage III — perspective correction, real-world measurement, and cropping.
#
# Two operating modes depending on CalibrationState:
#
#   Calibrated (ArUco found):
#     1. Apply homography stored in CalibrationState → rectified image
#     2. Transform detected boxes through the same homography
#     3. Measure bounding boxes in mm using the ArUco-derived mm_per_px
#
#   Approximate (no ArUco marker):
#     1. Skip perspective correction — use the original preprocessed image
#     2. Measure bounding boxes using the fallback mm_per_px constant
#     3. measurement_reliable = False is propagated to all CroppedContainer
#        objects so downstream stages and the UI can flag the degraded quality
# =============================================================================

from dataclasses import dataclass
import numpy as np
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from pipeline.stage2_detection import StageTwoResult, DetectedContainer
from cv_tasks.aruco import CalibrationState
from cv_tasks.homography import rectify_image, crop_box, measure_box_mm
import config as cfg


@dataclass
class CroppedContainer:
    crop_rgb:    np.ndarray         # crop of container interior
    box:         list               # [x_min, y_min, x_max, y_max] pixels
    label:       str                # "small container" / "large container"
    score:       float              # detection confidence from Stage II
    width_mm:    float              # measured width in mm
    height_mm:   float              # measured height in mm
    mm_per_px:   float              # scale factor used
    measurement_reliable: bool      # False in approximate mode


@dataclass
class StageThreeResult:
    crops:            list[CroppedContainer]
    rectified_rgb:    np.ndarray    # rectified or original image (for display)
    calibration:      CalibrationState


def run(stage2: StageTwoResult) -> StageThreeResult:
    """
    Execute Stage III.

    Parameters
    ----------
        stage2 : StageTwoResult from pipeline.stage2_detection

    Returns
    -------
        StageThreeResult
    """
    calibration = stage2.calibration
    mm_per_px = calibration["mm_per_px"]

    if calibration.aruco_detected:
        # --- Calibrated mode ---
        H = calibration.homography
        rectified = rectify_image(stage2.image_rgb, H)
        crops = _build_crops(
            stage2.containers,
            rectified,
            H,
            mm_per_px,
            measurement_reliable=True,
            apply_homography_to_boxes=True
        )
    else:
        # --- Approximate mode ---
        # No perspective correction; use the original preprocessed image.
        rectified = stage2.image_rgb.copy()
        crops = _build_crops(
            stage2.containers,
            rectified,
            None,
            mm_per_px,
            measurement_reliable=False,
            apply_homography_to_boxes=False
        )

    return StageThreeResult(
        crops = crops,
        rectified_rgb = rectified,
        calibration = calibration
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_crops(
    containers: list[DetectedContainer],
    image_rgb: np.ndarray,
    H,               # 3×3 homography or None
    mm_per_px: float,
    measurement_reliable: bool,
    apply_homography_to_boxes: bool
    ) -> list[CroppedContainer]:
    crops = []
    for container in containers:
        if apply_homography_to_boxes and H is not None:
            box_corners = _box_to_corners(container.box)
            box_processed = _transform_box(box_corners, H)
        else:
            box_processed = list(container.box)

        measurements = measure_box_mm(box_processed, mm_per_px)
        crop = crop_box(image_rgb, box_processed, padding_px=6)

        crops.append(
            CroppedContainer(
                crop_rgb             = crop,
                box                  = box_processed,
                label                = container.label,
                score                = container.score,
                width_mm             = measurements["width_mm"],
                height_mm            = measurements["height_mm"],
                mm_per_px            = mm_per_px,
                measurement_reliable = measurement_reliable
            )
        )
    return crops


def _box_to_corners(box: list) -> np.ndarray:
    """
    Convert [x_min, y_min, x_max, y_max] to 4 corner points (4x2).
    """
    x0, y0, x1, y1 = box
    return np.array(
        [
            [x0, y0], [x1, y0], [x1, y1], [x0, y1],
        ],
        dtype = np.float32
        )


def _transform_box(corners: np.ndarray, H: np.ndarray) -> list:
    """
    Apply homography H to the 4 corners of a bounding box and return
    the axis-aligned bounding box of the transformed corners.
    """
    ones = np.ones((4, 1), dtype = np.float32)
    pts = np.hstack([corners, ones])           # (4, 3)
    transformed = (H @ pts.T).T                # (4, 3)

    # Convert from homogeneous coordinates
    w    = transformed[:, 2:3]
    xy   = transformed[:, :2] / w
    x_min, y_min = xy.min(axis = 0)
    x_max, y_max = xy.max(axis = 0)
    return [float(x_min), float(y_min), float(x_max), float(y_max)]