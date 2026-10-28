# GN Food Estimator (local)

> For the project overview, architecture and results, see the [main README](../README.md).

Estimate food volume and CO₂ emissions from GN (Gastronorm) containers
using computer vision and deep learning.

Built by [Food Be Good](https://www.foodbegood.app/) as part of a food
waste reduction initiative.

This folder runs the Streamlit app locally on your computer.

## How it works

1. Take a top-down photo of your GN container with an ArUco marker visible
2. The app detects the container, estimates fill level, identifies the food,
   and calculates the associated CO₂ emissions
3. Take a second photo after food collection to see how much CO₂ was saved

## Models

- **Model #1** — Faster R-CNN ResNet-50 FPN: container detection
- **Model #2** — EfficientNet-B0: fill level classification. The app loads
  the weights-only `model2_small_state_dict.pth` / `model2_large_state_dict.pth`
  for the detected container size (falling back to the combined
  `model2_state_dict.pth`); set `MODEL2_PER_CONTAINER = False` in
  `src/config.py` to always use the combined model. The full fastai exports
  (`model2_fill*.pth`) are not used: they only load on the Python version
  they were exported with
- **Model #3** — nateraw/food (ViT, Food-101) fine-tuned on lentils / rice:
  food identification (`model3_food_id.pth` + `model3_label_map.json`). Set
  `MODEL3_FINETUNED = False` in `src/config.py` to use the stock 101-class
  nateraw/food instead

Models #1 and #2 are stored in `src/models/` (Git LFS — run `git lfs pull`
if the `.pth` files are only a few bytes). Model #3 is downloaded from the
HuggingFace Hub on the first run (~350 MB) and cached in
`src/models/nateraw_food_cache/` (git-ignored).

## Setup (Windows, PowerShell)

Requires Python 3.12. The virtual environment lives at the repository root
in `.venv/` (git-ignored).

```powershell
# From the repository root
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip

# NVIDIA GPU: install the CUDA build of PyTorch first
pip install torch==2.10.0 torchvision --index-url https://download.pytorch.org/whl/cu126
# (CPU only: skip the line above — requirements.txt installs the CPU build)

pip install -r streamlit_app_local\requirements.txt
```

In VS Code, select `.venv\Scripts\python.exe` via **Python: Select Interpreter**.

## Run

Run from inside `src/` so Streamlit picks up the theme in
`src/.streamlit/config.toml`:

```powershell
.venv\Scripts\Activate.ps1
cd streamlit_app_local\src
streamlit run app.py
```

The app opens at <http://localhost:8501>.

## Notes

- An ArUco marker (DICT_4X4_50, ID 0, 94 mm side) must be visible in every photo
- Supported foods: rice, lentils
- CO₂ estimates based on German/Bavarian lifecycle data (ifeu 2020, KTBL)
- Set `STUB_MODE = True` in `src/config.py` to run the UI with dummy data
  (no models needed)
