# =============================================================================
# pipeline/stage8_mass.py
# =============================================================================
# Stage VIII — estimate food mass in kg.
#
#   mass_kg = food_volume_l × density_kg_per_l
# =============================================================================

from dataclasses import dataclass
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from pipeline.stage7_food_id import StageSevenResult, FoodIdResult
import config as cfg


@dataclass
class MassResult:
    food_type:       str
    food_confidence: float
    ambiguous:       bool
    gn_id:           str
    gn_label:        str
    container_vol_l: float
    fill_ratio:      float
    fill_label:      str
    fill_confidence: float
    low_confidence:  bool
    food_volume_l:   float
    density_kg_l:    float    # kg per litre
    mass_kg:         float    # food_volume_l × density_kg_l
    container_label: str
    detection_score: float
    snap_warning:    bool
    top_k:           list


@dataclass
class StageEightResult:
    masses:        list[MassResult]
    aruco:         dict
    rectified_rgb: object


def run(stage7: StageSevenResult) -> StageEightResult:
    """
    Execute Stage VIII.
    """
    results = []
    for fi in stage7.food_ids:
        density = cfg.FOOD_DENSITIES.get(fi.food_type, 0.0)
        mass_kg = fi.food_volume_l * density

        results.append(
            MassResult(
                food_type = fi.food_type,
                food_confidence = fi.food_confidence,
                ambiguous = fi.ambiguous,
                gn_id = fi.gn_id,
                gn_label = fi.gn_label,
                container_vol_l = fi.container_vol_l,
                fill_ratio = fi.fill_ratio,
                fill_label = fi.fill_label,
                fill_confidence = fi.fill_confidence,
                low_confidence = fi.low_confidence,
                food_volume_l = fi.food_volume_l,
                density_kg_l = density,
                mass_kg = mass_kg,
                container_label = fi.container_label,
                detection_score = fi.detection_score,
                snap_warning = fi.snap_warning,
                top_k = fi.top_k
        ))

    return StageEightResult(
        masses = results,
        aruco = stage7.aruco,
        rectified_rgb = stage7.rectified_rgb
    )