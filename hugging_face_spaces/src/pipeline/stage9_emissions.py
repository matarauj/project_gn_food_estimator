# =============================================================================
# pipeline/stage9_emissions.py
# =============================================================================
# Stage IX — estimate CO₂ emissions.
#
#   co2_kg = mass_kg × emission_factor_kg_co2_per_kg
# =============================================================================

from dataclasses import dataclass
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from pipeline.stage8_mass import StageEightResult, MassResult
from cv_tasks.aruco import CalibrationState
import config as cfg


@dataclass
class EmissionResult:
    food_type:            str
    food_confidence:      float
    ambiguous:            bool
    gn_id:                str
    gn_label:             str
    container_vol_l:      float
    fill_ratio:           float
    fill_label:           str
    fill_confidence:      float
    low_confidence:       bool
    food_volume_l:        float
    density_kg_l:         float
    mass_kg:              float
    emission_factor:      float    # kg CO₂e per kg food
    co2_kg:               float    # mass_kg × emission_factor
    container_label:      str
    detection_score:      float
    snap_warning:         bool
    measurement_reliable: bool
    top_k:                list


@dataclass
class StageNineResult:
    emissions:      list[EmissionResult]
    total_co2_kg:   float       # sum across all containers in this photo
    calibration:    CalibrationState
    rectified_rgb:  object


def run(stage8: StageEightResult) -> StageNineResult:
    """
    Execute Stage IX.
    """
    results = []
    for m in stage8.masses:
        factor = cfg.CO2_EMISSION_FACTORS.get(m.food_type, 0.0)
        co2_kg = m.mass_kg * factor

        results.append(
            EmissionResult(
                food_type       = m.food_type,
                food_confidence = m.food_confidence,
                ambiguous       = m.ambiguous,
                gn_id           = m.gn_id,
                gn_label        = m.gn_label,
                container_vol_l = m.container_vol_l,
                fill_ratio      = m.fill_ratio,
                fill_label      = m.fill_label,
                fill_confidence = m.fill_confidence,
                low_confidence  = m.low_confidence,
                food_volume_l   = m.food_volume_l,
                density_kg_l    = m.density_kg_l,
                mass_kg         = m.mass_kg,
                emission_factor = factor,
                co2_kg          = co2_kg,
                container_label = m.container_label,
                detection_score = m.detection_score,
                snap_warning    = m.snap_warning,
                measurement_reliable = m.measurement_reliable,
                top_k           = m.top_k
            )
        )

    total_co2 = sum(r.co2_kg for r in results)

    return StageNineResult(
        emissions = results,
        total_co2_kg = total_co2,
        calibration   = stage8.calibration,
        rectified_rgb = stage8.rectified_rgb
    )