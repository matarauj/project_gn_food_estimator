# =============================================================================
# pipeline/stage3_geometry.py
# =============================================================================
# Stage III — perspective correction, real-world measurement, and cropping.
#
# For each detected container:
#   1. Compute and apply homography using ArUco corners
#   2. Measure the container bounding box in mm using the ArUco scale
#   3. Crop the rectified image to the container bounding box
# =============================================================================

from dataclasses import dataclass, field
import numpy as np
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from pipeline.stage2_detection import StageTwoResult, DetectedContainer
from cv_tasks.homography import compute_homography, rectify_image, crop_box, measure_box_mm
import config as cfg


@dataclass
class CroppedContainer:
    crop_rgb:    np.ndarray     # rectified crop of container interior
    box:         list           # [x_min, y_min, x_max, y_max] pixels (rectified)
    label:       str            # "small container" / "large container"
    score:       float          # detection confidence from Stage II
    width_mm:    float          # measured width in mm
    height_mm:   float          # measured height in mm
    mm_per_px:   float          # scale factor used


@dataclass
class StageThreeResult:
    crops:            list[CroppedContainer]
    rectified_rgb:    np.ndarray    # full rectified image (for display)
    aruco:            dict


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
    aruco      = stage2.aruco
    mm_per_px  = aruco["mm_per_px"]
    H          = compute_homography(aruco["corners"], mm_per_px = mm_per_px)
    rectified  = rectify_image(stage2.image_rgb, H)

    crops = []
    for container in stage2.containers:
        # Transform the detected box corners through H
        box_corners = _box_to_corners(container.box)
        box_rectified = _transform_box(box_corners, H)

        measurements = measure_box_mm(box_rectified, mm_per_px)
        crop = crop_box(rectified, box_rectified, padding_px=6)

        crops.append(
            CroppedContainer(
                crop_rgb = crop,
                box = box_rectified,
                label = container.label,
                score = container.score,
                width_mm = measurements["width_mm"],
                height_mm = measurements["height_mm"],
                mm_per_px = mm_per_px
            )
        )
        
    return StageThreeResult(
        crops = crops,
        rectified_rgb = rectified,
        aruco = aruco
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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
    pts  = np.hstack([corners, ones])          # (4, 3)
    transformed = (H @ pts.T).T                # (4, 3)

    # Convert from homogeneous coordinates
    w    = transformed[:, 2:3]
    xy   = transformed[:, :2] / w
    x_min, y_min = xy.min(axis = 0)
    x_max, y_max = xy.max(axis = 0)
    return [float(x_min), float(y_min), float(x_max), float(y_max)]