# =============================================================================
# pipeline/stage10_11_compare.py
# =============================================================================
# Stage X  — run the full pipeline (I–IX) on the second photo.
# Stage XI — compute CO₂ saved = CO₂_before − CO₂_after.
#
# Stage X is not a separate function here — the app runs the full pipeline
# on photo 2 using the same stage functions used for photo 1.
# This module only contains the Stage XI comparison logic.
# =============================================================================

from dataclasses import dataclass
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from pipeline.stage9_emissions import StageNineResult, EmissionResult


@dataclass
class ContainerComparison:
    """
    Side-by-side comparison for one matched container pair.
    """
    container_label:  str
    gn_id:            str
    food_type_before: str
    food_type_after:  str
    mass_kg_before:   float
    mass_kg_after:    float
    co2_kg_before:    float
    co2_kg_after:     float
    co2_saved_kg:     float     # positive = reduction, negative = increase
    fill_before:      str
    fill_after:       str


@dataclass
class ComparisonResult:
    containers:         list[ContainerComparison]
    total_co2_before:   float
    total_co2_after:    float
    total_co2_saved:    float    # total_co2_before − total_co2_after
    total_mass_before:  float
    total_mass_after:   float


def run(stage9_before: StageNineResult,
        stage9_after:  StageNineResult) -> ComparisonResult:
    """
    Execute Stage XI.

    Matches containers between the two photos by position in the results list.
    If photo 2 has fewer containers than photo 1, unmatched containers are
    assumed to be empty (mass = 0, CO₂ = 0).

    Parameters
    ----------
    stage9_before : StageNineResult — emissions from photo 1
    stage9_after  : StageNineResult — emissions from photo 2

    Returns
    -------
    ComparisonResult
    """
    before_list = stage9_before.emissions
    after_list  = stage9_after.emissions

    # Pad the shorter list with zero-emission placeholders
    n = max(len(before_list), len(after_list))
    before_list = _pad(before_list, n)
    after_list  = _pad(after_list,  n)

    comparisons = []
    for b, a in zip(before_list, after_list):
        comparisons.append(
            ContainerComparison(
                container_label  = b.container_label if b else a.container_label,
                gn_id            = b.gn_id if b else a.gn_id,
                food_type_before = b.food_type if b else "—",
                food_type_after  = a.food_type if a else "—",
                mass_kg_before   = b.mass_kg   if b else 0.0,
                mass_kg_after    = a.mass_kg   if a else 0.0,
                co2_kg_before    = b.co2_kg    if b else 0.0,
                co2_kg_after     = a.co2_kg    if a else 0.0,
                co2_saved_kg     = (b.co2_kg if b else 0.0) - (a.co2_kg if a else 0.0),
                fill_before      = b.fill_label if b else "—",
                fill_after       = a.fill_label if a else "—"
        ))

    total_before = sum(c.co2_kg_before for c in comparisons)
    total_after  = sum(c.co2_kg_after  for c in comparisons)
    mass_before  = sum(c.mass_kg_before for c in comparisons)
    mass_after   = sum(c.mass_kg_after  for c in comparisons)

    return ComparisonResult(
        containers = comparisons,
        total_co2_before = total_before,
        total_co2_after = total_after,
        total_co2_saved = total_before - total_after,
        total_mass_before = mass_before,
        total_mass_after = mass_after
    )


def _pad(lst: list, n: int) -> list:
    """Pad a list to length n with None."""
    return lst + [None] * (n - len(lst))