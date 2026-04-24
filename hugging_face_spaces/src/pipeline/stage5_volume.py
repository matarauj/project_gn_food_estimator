# =============================================================================
# pipeline/stage5_volume.py
# =============================================================================
# Stage V — estimate the total volume of each container.
#
# Combines:
#   - The container label from Stage II ("small container" / "large container")
#   - The mm measurements from Stage III (width_mm, height_mm)
#
# Strategy:
#   1. Use the label from Model #1 as the primary identifier.
#   2. Compute a pixel-derived volume from the measured dimensions and the
#      known container depth (from config.GN_CONTAINERS).
#   3. If the pixel-derived volume is within GN_SNAP_TOLERANCE of a standard
#      GN volume, snap to that standard — this corrects small measurement
#      errors and guarantees we always output a known GN size.
#   4. If the pixel-derived volume disagrees with the label-based volume by
#      more than the tolerance, prefer the label (Model #1 is more reliable
#      than pixel measurement at this dataset size) and log a warning.
# =============================================================================

from dataclasses import dataclass
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from pipeline.stage4_fill_level import StageFourResult, FillResult
import config as cfg


@dataclass
class VolumeResult:
    gn_id:            str      # e.g. "GN_1_1"
    gn_label:         str      # e.g. "large container"
    volume_l:         float    # total container volume in litres
    inner_l_mm:       float    # standard inner length (mm)
    inner_w_mm:       float    # standard inner width (mm)
    depth_mm:         float    # standard depth (mm)
    measured_w_mm:    float    # measured width from Stage III
    measured_h_mm:    float    # measured height from Stage III
    snapped:          bool     # True if measurement was snapped to standard
    snap_warning:     bool     # True if label and measurement disagreed
    # Passed through from Stage IV
    fill_ratio:       float
    fill_label:       str
    fill_confidence:  float
    low_confidence:   bool
    container_label:  str
    detection_score:  float


@dataclass
class StageFiveResult:
    volumes:        list[VolumeResult]
    aruco:          dict
    rectified_rgb:  object   # np.ndarray, passed through for display


def run(stage4: StageFourResult) -> StageFiveResult:
    """
    Execute Stage V.
    """
    volumes = []
    for fill in stage4.fills:
        vr = _estimate_volume(fill)
        volumes.append(vr)
    return StageFiveResult(
        volumes = volumes,
        aruco = stage4.aruco,
        rectified_rgb = stage4.rectified_rgb,
    )


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------

def _label_to_gn_id(label: str) -> str | None:
    """
    Map a Model #1 label to a GN container ID.
    """
    for gn_id, spec in cfg.GN_CONTAINERS.items():
        if spec["label"] == label:
            return gn_id
    return None


def _snap_to_standard(measured_volume_l: float) -> tuple[str, bool]:
    """
    Find the nearest GN standard volume within GN_SNAP_TOLERANCE.

    Returns (gn_id, snapped) where snapped=True if a match was found.
    If no standard is within tolerance, returns the nearest one anyway
    with snapped=False (caller decides what to do).
    """
    best_id    = None
    best_delta = float("inf")
    for gn_id, spec in cfg.GN_CONTAINERS.items():
        delta = abs(measured_volume_l - spec["volume_l"]) / spec["volume_l"]
        if delta < best_delta:
            best_delta = delta
            best_id    = gn_id
    snapped = best_delta <= cfg.GN_SNAP_TOLERANCE
    return best_id, snapped


def _estimate_volume(fill: FillResult) -> VolumeResult:
    # Step 1: label-based GN ID
    label_gn_id = _label_to_gn_id(fill.container_label)

    # Step 2: pixel-derived volume
    # Use measured width and height + known depth from the label-matched spec
    # (we cannot measure depth from a top-down photo)
    if label_gn_id:
        depth_mm = cfg.GN_CONTAINERS[label_gn_id]["depth_mm"]
    else:
        # Fallback: use the first container's depth
        depth_mm = list(cfg.GN_CONTAINERS.values())[0]["depth_mm"]

    # Volume in Litre: 1e-6 | in Cubic meter: 1e-9
    measured_vol_l = (
        fill.width_mm * fill.height_mm * depth_mm / 1e6
    )

    # Step 3: snap measured volume to nearest standard
    snap_gn_id, snapped = _snap_to_standard(measured_vol_l)

    # Step 4: resolve disagreement between label and snap
    snap_warning = False
    if label_gn_id != snap_gn_id:
        snap_warning = True
        # Prefer the label from Model #1 — more reliable than pixel measurement
        final_gn_id = label_gn_id
    else:
        final_gn_id = snap_gn_id

    spec = cfg.GN_CONTAINERS[final_gn_id]

    return VolumeResult(
        gn_id =             final_gn_id,
        gn_label =          spec["label"],
        volume_l =          spec["volume_l"],
        inner_l_mm =        spec["inner_l_mm"],
        inner_w_mm =        spec["inner_w_mm"],
        depth_mm =          spec["depth_mm"],
        measured_w_mm =     fill.width_mm,
        measured_h_mm =     fill.height_mm,
        snapped =           snapped,
        snap_warning =      snap_warning,
        fill_ratio =        fill.fill_ratio,
        fill_label =        fill.label,
        fill_confidence =   fill.confidence,
        low_confidence =    fill.low_confidence,
        container_label =   fill.container_label,
        detection_score =   fill.detection_score
    )