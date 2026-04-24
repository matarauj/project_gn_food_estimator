# =============================================================================
# pipeline/stage6_food_volume.py
# =============================================================================
# Stage VI — estimate the current volume of food in each container.
#
#   food_volume_l = container_volume_l × fill_ratio
# =============================================================================

from dataclasses import dataclass
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from pipeline.stage5_volume import StageFiveResult, VolumeResult


@dataclass
class FoodVolumeResult:
    gn_id:           str
    gn_label:        str
    container_vol_l: float   # total container volume
    fill_ratio:      float
    fill_label:      str
    fill_confidence: float
    low_confidence:  bool
    food_volume_l:   float   # container_vol_l × fill_ratio
    # Geometry — passed through for Stage VIII
    inner_l_mm:      float
    inner_w_mm:      float
    depth_mm:        float
    measured_w_mm:   float
    measured_h_mm:   float
    snap_warning:    bool
    detection_score: float
    container_label: str


@dataclass
class StageSixResult:
    food_volumes:  list[FoodVolumeResult]
    aruco:         dict
    rectified_rgb: object


def run(stage5: StageFiveResult) -> StageSixResult:
    """
    Execute Stage VI.
    """
    results = []
    for vol in stage5.volumes:
        food_vol = vol.container_vol_l if not hasattr(vol, 'volume_l') else vol.volume_l
        food_vol = vol.volume_l * vol.fill_ratio

        results.append(
            FoodVolumeResult(
                gn_id = vol.gn_id,
                gn_label = vol.gn_label,
                container_vol_l = vol.volume_l,
                fill_ratio = vol.fill_ratio,
                fill_label = vol.fill_label,
                fill_confidence = vol.fill_confidence,
                low_confidence = vol.low_confidence,
                food_volume_l = food_vol,
                inner_l_mm = vol.inner_l_mm,
                inner_w_mm = vol.inner_w_mm,
                depth_mm = vol.depth_mm,
                measured_w_mm = vol.measured_w_mm,
                measured_h_mm = vol.measured_h_mm,
                snap_warning = vol.snap_warning,
                detection_score = vol.detection_score,
                container_label = vol.container_label
        ))

    return StageSixResult(
        food_volumes = results,
        aruco = stage5.aruco,
        rectified_rgb = stage5.rectified_rgb
    )