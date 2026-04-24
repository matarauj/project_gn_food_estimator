# =============================================================================
# pipeline/stage4_fill_level.py
# =============================================================================
# Stage IV — classify the fill level of each cropped container.
#
# Uses Model #2 on each crop from Stage III.
# If confidence is below cfg.CONFIDENCE_THRESHOLD, sets low_confidence=True
# so the app can prompt the user to retake the photo.
# =============================================================================

from dataclasses import dataclass
import numpy as np
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from pipeline.stage3_geometry import StageThreeResult, CroppedContainer
from cv_tasks.model_loader import get_fill_classifier
import config as cfg

_classifier = None


def _get_classifier():
    global _classifier
    if _classifier is None:
        _classifier = get_fill_classifier()
    return _classifier


@dataclass
class FillResult:
    label:          str         # "empty" | "low" | "medium" | "high" | "full"
    confidence:     float       # max softmax probability
    probs:          dict        # {class: probability} for all 5 levels
    fill_ratio:     float       # numeric ratio from cfg.FILL_RATIOS
    low_confidence: bool        # True if below cfg.CONFIDENCE_THRESHOLD
    crop_rgb:       np.ndarray  # the container crop (passed through for display)
    box:            list        # bounding box (passed through)
    container_label: str        # "small container" / "large container"
    width_mm:       float
    height_mm:      float
    mm_per_px:      float
    detection_score: float


@dataclass
class StageFourResult:
    fills:     list[FillResult]
    aruco:     dict
    rectified_rgb: np.ndarray


def run(stage3: StageThreeResult) -> StageFourResult:
    """
    Execute Stage IV.

    Parameters
    ----------
    stage3 : StageThreeResult from pipeline.stage3_geometry

    Returns
    -------
    StageFourResult
    """
    classifier = _get_classifier()
    fills = []

    for crop_container in stage3.crops:
        pred = classifier.predict(crop_container.crop_rgb)

        fill_ratio     = cfg.FILL_RATIOS.get(pred["label"], 0.0)
        low_confidence = pred["confidence"] < cfg.CONFIDENCE_THRESHOLD

        fills.append(
            FillResult(
                label = pred["label"],
                confidence = pred["confidence"],
                probs = pred["probs"],
                fill_ratio = fill_ratio,
                low_confidence = low_confidence,
                crop_rgb = crop_container.crop_rgb,
                box = crop_container.box,
                container_label = crop_container.label,
                width_mm = crop_container.width_mm,
                height_mm = crop_container.height_mm,
                mm_per_px = crop_container.mm_per_px,
                detection_score = crop_container.score
            )
        )

    return StageFourResult(
        fills = fills,
        aruco = stage3.aruco,
        rectified_rgb = stage3.rectified_rgb
    )