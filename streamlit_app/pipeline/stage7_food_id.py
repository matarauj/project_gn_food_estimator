# =============================================================================
# pipeline/stage7_food_id.py
# =============================================================================
# Stage VII — identify the food type in each container crop.
#
# Model #3: nateraw/food (ViT-base, Food-101, HuggingFace)
# https://huggingface.co/nateraw/food
#
# Loading strategy:
#   - On first run the model is downloaded from HuggingFace Hub (~350 MB)
#     and saved to cfg.MODEL3_CACHE_DIR (models/nateraw_food_cache/).
#   - On subsequent runs it loads directly from the cache — no internet
#     connection needed and no re-download.
#   - The pipeline is a module-level singleton so it is loaded once per
#     app session, not once per photo.
#
# Manual override:
#   If the top prediction score is below cfg.MODEL3_MIN_SCORE, or if the
#   top class cannot be mapped to a supported food, food_type is set to
#   "unknown" and ambiguous=True.
#   When ambiguous=True the app shows a selectbox so the user can correct
#   the food type. The override is passed back into this module via
#   run_with_overrides(), which replaces the ambiguous prediction with
#   the user's choice before the result propagates to Stage VIII.
#
# In STUB_MODE, returns a cycling sequence of supported food types.
# In live mode, runs the HuggingFace pipeline and maps Food-101 class names
# to our canonical food names via config.FOOD_CLASS_MAP.
# =============================================================================

from dataclasses import dataclass
import numpy as np
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from pipeline.stage6_food_volume import StageSixResult, FoodVolumeResult
import config as cfg


_hf_pipeline = None    # module-level singleton


def _get_hf_pipeline():
    """
    Load the HuggingFace pipeline, using the local cache if available.
    Downloads on first call, then loads from disk on all subsequent calls.
    """
    global _hf_pipeline
    if _hf_pipeline is not None:
        return _hf_pipeline
 
    from transformers import (
        pipeline as hf_pipeline_fn,
        AutoModelForImageClassification,
        AutoImageProcessor
    )
 
    cache_dir = cfg.MODEL3_CACHE_DIR
    cache_dir.mkdir(parents = True, exist_ok = True)
    cache_str = str(cache_dir)
 
    # Check whether the model is already cached
    cached_files = list(cache_dir.rglob("config.json"))
    if cached_files:
        print(f"[Model #3] Loading from local cache: {cache_dir}")
    else:
        print(f"[Model #3] Downloading {cfg.MODEL3_HF_NAME} "
              f"(~350 MB, first run only)…")
    
    # Load model and feature extractor explicitly so cache_dir is honoured.
    model = AutoModelForImageClassification.from_pretrained(
        cfg.MODEL3_HF_NAME,
        cache_dir = cache_str
    )
    image_processor = AutoImageProcessor.from_pretrained(
        cfg.MODEL3_HF_NAME,
        cache_dir = cache_str
    )

    _hf_pipeline = hf_pipeline_fn(
        "image-classification",
        model = model,
        image_processor = image_processor, 
        top_k = cfg.MODEL3_TOP_K
    )
    print("[Model #3] Ready.")
    return _hf_pipeline


# ---------------------------------------------------------------------------
# Stub food identifier
# ---------------------------------------------------------------------------

class _StubFoodIdentifier:
    """
    Cycles through supported foods for UI testing.
    """
    _foods = ["rice", "lentils"]
    _idx   = 0

    def identify(self, image_rgb: np.ndarray) -> dict:
        food = self._foods[_StubFoodIdentifier._idx % len(self._foods)]
        _StubFoodIdentifier._idx += 1
        return {
            "food_type": food,
            "confidence": 0.85,
            "top_k": [{
                "label": food, "mapped": food, "score": 0.85
            }],
            "ambiguous": False
        }


# ---------------------------------------------------------------------------
# Real food identifier
# ---------------------------------------------------------------------------

class _HFFoodIdentifier:
    """
    Runs nateraw/food and maps Food-101 labels to canonical food names.
    """
    def identify(self, image_rgb: np.ndarray) -> dict:
        from PIL import Image as PILImage
        
        pipe   = _get_hf_pipeline()
        pil    = PILImage.fromarray(image_rgb)
        preds  = pipe(pil)   # list of {label, score}

        # Normalise label format and apply FOOD_CLASS_MAP
        top_k = []
        for p in preds:
            norm   = p["label"].lower().replace(" ", "_")
            mapped = cfg.FOOD_CLASS_MAP.get(norm)
            top_k.append({
                "label":  p["label"],
                "mapped": mapped,
                "score":  p["score"]
            })
 
        # Best prediction that maps to a supported food and clears threshold
        best = next(
            (p for p in top_k
             if p["mapped"] in cfg.SUPPORTED_FOODS
             and p["score"] >= cfg.MODEL3_MIN_SCORE),
            None
        )
 
        if best is None:
            return {
                "food_type":  "unknown",
                "confidence": preds[0]["score"] if preds else 0.0,
                "top_k":      top_k,
                "ambiguous":  True
            }
 
        return {
            "food_type":  best["mapped"],
            "confidence": best["score"],
            "top_k":      top_k,
            "ambiguous":  False
        }


def _get_identifier():
    if cfg.STUB_MODE:
        return _StubFoodIdentifier()
        #return _HFFoodIdentifier()
    return _HFFoodIdentifier()


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class FoodIdResult:
    food_type:        str      # "rice" | "lentils" | "unknown"
    food_confidence:  float
    top_k:            list     # raw top-k predictions for display
    ambiguous:        bool     # True → show manual override UI
    override_applied: bool     # True → user corrected the prediction
    # Passed through from Stage VI
    gn_id:            str
    gn_label:         str
    container_vol_l:  float
    fill_ratio:       float
    fill_label:       str
    fill_confidence:  float
    low_confidence:   bool
    food_volume_l:    float
    container_label:  str
    detection_score:  float
    snap_warning:     bool


@dataclass
class StageSevenResult:
    food_ids:      list[FoodIdResult]
    aruco:         dict
    rectified_rgb: object


def run(stage6: StageSixResult,
        crop_images: list[np.ndarray]) -> StageSevenResult:
    """
    Execute Stage VII.
    At the moment, the system uses an image classifier, rather than an instance segmentation.

    Parameters
    ----------
    stage6       : StageSixResult
    crop_images  : list[np.ndarray]
        One RGB crop per container, sourced from stage4.fills[i].crop_rgb

    Returns
    -------
    StageSevenResult
        Items with ambiguous=True require a manual override before Stage VIII.
    """
    identifier = _get_identifier()
    results    = []

    for fv, crop in zip(stage6.food_volumes, crop_images):
        pred = identifier.identify(crop)
        results.append(_make_result(pred, fv, override_applied = False))

    return StageSevenResult(
        food_ids = results,
        aruco = stage6.aruco,
        rectified_rgb = stage6.rectified_rgb
    )


def apply_overrides(stage7: StageSevenResult,
                    overrides: dict[int, str]) -> StageSevenResult:
    """
    Apply manual food type overrides chosen by the user in the app UI.
 
    Parameters
    ----------
    stage7    : StageSevenResult — result from run()
    overrides : dict[int, str]
        Keys are container indices (0-based), values are food type strings
        chosen by the user (must be in cfg.SUPPORTED_FOODS).
 
    Returns
    -------
    A new StageSevenResult with the overridden items updated.
    """
    updated = []
    for i, fi in enumerate(stage7.food_ids):
        if i in overrides:
            food = overrides[i]
            updated.append(
                    FoodIdResult(
                        food_type = food,
                        food_confidence = 1.0,    # user-confirmed → treat as certain
                        top_k = fi.top_k,
                        ambiguous = False,
                        override_applied = True,
                        gn_id = fi.gn_id,
                        gn_label = fi.gn_label,
                        container_vol_l = fi.container_vol_l,
                        fill_ratio = fi.fill_ratio,
                        fill_label = fi.fill_label,
                        fill_confidence = fi.fill_confidence,
                        low_confidence = fi.low_confidence,
                        food_volume_l = fi.food_volume_l,
                        container_label = fi.container_label,
                        detection_score = fi.detection_score,
                        snap_warning = fi.snap_warning
                    )
                )
        else:
            updated.append(fi)
 
    return StageSevenResult(
        food_ids = updated,
        aruco = stage7.aruco,
        rectified_rgb = stage7.rectified_rgb
    )
 
 
def _make_result(
        pred: dict,
        fv: FoodVolumeResult,
        override_applied: bool) -> FoodIdResult:
    return FoodIdResult(
        food_type = pred["food_type"],
        food_confidence = pred["confidence"],
        top_k = pred["top_k"],
        ambiguous = pred["ambiguous"],
        override_applied = override_applied,
        gn_id = fv.gn_id,
        gn_label = fv.gn_label,
        container_vol_l = fv.container_vol_l,
        fill_ratio = fv.fill_ratio,
        fill_label = fv.fill_label,
        fill_confidence = fv.fill_confidence,
        low_confidence = fv.low_confidence,
        food_volume_l = fv.food_volume_l,
        container_label = fv.container_label,
        detection_score = fv.detection_score,
        snap_warning = fv.snap_warning
    )
