# =============================================================================
# pipeline/stage2_detection.py
# =============================================================================
# Stage II — detect GN containers using Model #1.
#
# Returns a list of DetectedContainer objects, one per detected box.
# Raises ContainerNotFoundError if no containers are detected above the
# confidence threshold — the app catches this and prompts the user.
# =============================================================================

from dataclasses import dataclass
import numpy as np
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from pipeline.stage1_capture import StageOneResult
from cv_tasks.aruco import CalibrationState
from cv_tasks.model_loader import get_detector
import config as cfg

# Module-level singleton so the model is loaded once per app session
_detector = None


def _get_detector():
    global _detector
    if _detector is None:
        _detector = get_detector()
    return _detector


class ContainerNotFoundError(Exception):
    """
    Raised when no GN containers are detected in the image.
    """
    pass


@dataclass
class DetectedContainer:
    box:   list        # [x_min, y_min, x_max, y_max] in pixels
    label: str         # "small container" or "large container"
    score: float       # detection confidence


@dataclass
class StageTwoResult:
    containers: list[DetectedContainer]
    image_rgb:  np.ndarray    # passed through from Stage I (unchanged)
    calibration: CalibrationState # passed through from Stage I


def run(stage1: StageOneResult) -> StageTwoResult:
    """
    Execute Stage II.

    Parameters
    ----------
    stage1 : StageOneResult from pipeline.stage1_capture

    Returns
    -------
    StageTwoResult

    Raises
    ------
    ContainerNotFoundError
        If no containers are detected above cfg.MODEL1_SCORE_THRESH.
    """
    detector = _get_detector()
    raw_preds = detector.predict(stage1.image_rgb)

    # Filter to container classes only (ignore any stray detections)
    valid_labels = set(cfg.MODEL1_CLASSES)
    containers = [
        DetectedContainer(box = p["box"], label = p["label"], score = p["score"]) 
        for p in raw_preds 
        if p["label"] in valid_labels
        ]

    if not containers:
        raise ContainerNotFoundError(
            "No GN containers detected in the image. "
            "Please ensure the container is clearly visible and retake the photo."
        )

    # Sort by confidence descending
    containers.sort(key = lambda c: c.score, reverse = True)

    return StageTwoResult(
        containers = containers,
        image_rgb = stage1.image_rgb,
        calibration = stage1.calibration
    )