# =============================================================================
# config.py — inference-only configuration for the GN Food Estimator app
# =============================================================================
from pathlib import Path

ROOT       = Path(__file__).resolve().parent
MODELS_DIR = ROOT / "models"


# -----------------------------------------------------------------------------
# Model paths
# -----------------------------------------------------------------------------
MODEL1_PATH     = MODELS_DIR / "model1_detector.pth"
MODEL1_LABEL_MAP = MODELS_DIR / "model1_label_map.json"
MODEL2_PATH     = MODELS_DIR / "model2_fill.pth"


# -----------------------------------------------------------------------------
# ArUco
# -----------------------------------------------------------------------------
ARUCO_DICT_ID   = "DICT_4X4_50"
ARUCO_MARKER_ID = 0
ARUCO_MARKER_MM = 94.0          # physical side length of printed marker (mm)


# -----------------------------------------------------------------------------
# Image
# -----------------------------------------------------------------------------
IMAGE_SIZE = 512
HIST_EQ_ENABLED = True      # Apply CLAHE histogram equalisation in Stage I


# -----------------------------------------------------------------------------
# GN container standards (EN 631-1)
# volume_l = inner_l × inner_w × depth / 1e6 | Volume in Litre: 1e-6 | in Cubic meter: 1e-9
# -----------------------------------------------------------------------------
GN_CONTAINERS = {
    "GN_1_1": {
        "label": "large container",
        "inner_l_mm": 228,
        "inner_w_mm": 128,
        "depth_mm": 65,
        "volume_l": round(228 * 128 * 65 / 1e6, 2)
    },
    "GN_1_2": {
        "label": "small container",
        "inner_l_mm": 117,
        "inner_w_mm": 98,
        "depth_mm": 40,
        "volume_l": round(117 * 98 * 40 / 1e6, 2)
    }
}
GN_SNAP_TOLERANCE = 0.10


# -----------------------------------------------------------------------------
# Model #1 — detection
# -----------------------------------------------------------------------------
MODEL1_CLASSES    = ["small container", "large container"]
MODEL1_SCORE_THRESH = 0.40    # boxes below this confidence are discarded
MODEL1_NMS_THRESH   = 0.30    # IoU threshold for non-maximum suppression


# -----------------------------------------------------------------------------
# Model #2 — fill level
# -----------------------------------------------------------------------------
MODEL2_CLASSES = ["empty", "low", "medium", "high", "full"]
FILL_RATIOS = {
    "empty": 0.00,
    "low": 0.20,
    "medium": 0.50,
    "high": 0.75,
    "full": 0.95
}
CONFIDENCE_THRESHOLD = 0.70   # below this → ask user to retake photo


# -----------------------------------------------------------------------------
# Food & emissions
# -----------------------------------------------------------------------------
FOOD_DENSITIES = {
    "rice": 0.85,
    "lentils": 0.90
    }

CO2_EMISSION_FACTORS = {
    "rice": 2.7,
    "lentils": 0.9
    }

CO2_DISCLAIMER = (
    "CO\u2082 estimates are approximate and based on average German/Bavarian "
    "lifecycle data (ifeu 2020, KTBL). Actual emissions vary by supplier, "
    "season, and production method. This tool is intended for indicative "
    "purposes only."
)


# -----------------------------------------------------------------------------
# Model #3 — food identification (HuggingFace nateraw/food)
# -----------------------------------------------------------------------------
MODEL3_HF_NAME    = "nateraw/food"                      # HuggingFace model identifier
MODEL3_CACHE_DIR = MODELS_DIR / "nateraw_food_cache"    # local cache folder
MODEL3_TOP_K      = 3                                   # return top-k food predictions
MODEL3_MIN_SCORE  = 0.30                                # below this → label as "unknown"
 
# Map HuggingFace Food-101 class names → our canonical food names
# Only classes relevant to your menu need entries here.
FOOD_CLASS_MAP = {
    "rice":         "rice",
    "fried_rice":   "rice",
    "risotto":      "rice",
    "lentil_soup":  "lentils",
    "lentils":      "lentils"
}

# Supported food types (must match keys in FOOD_DENSITIES and CO2_EMISSION_FACTORS)
SUPPORTED_FOODS = list(FOOD_DENSITIES.keys())   # ["rice", "lentils"]


# -----------------------------------------------------------------------------
# Stub mode — set True while .pth files are not yet available
# All pipeline stages return realistic dummy data instead of running models.
# Flip to False once model files are placed in models/
# -----------------------------------------------------------------------------
STUB_MODE = True