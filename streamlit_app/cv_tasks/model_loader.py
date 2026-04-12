# =============================================================================
# cv_tasks/model_loader.py
# =============================================================================
# Loads Model #1 (Faster R-CNN) and Model #2 (EfficientNet-B0) for inference.
# When config.STUB_MODE = True, returns lightweight stub objects that return
# realistic dummy data so the UI can be developed before training complets.
# =============================================================================

import json
import numpy as np
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))
import config as cfg


# =============================================================================
# Stub classes
# =============================================================================

class StubDetector:
    """
    Returns a single plausible bounding box centred in the image.
    Mimics the output format of the real Faster R-CNN inference function.
    """
    def predict(self, image_rgb: np.ndarray) -> list[dict]:
        h, w = image_rgb.shape[:2]
        margin = 0.15
        box = [w*margin, h*margin, w*(1-margin), h*(1-margin)]
        return [{
            "box":   box,
            "label": "large container",
            "score": 0.91
        }]


class StubFillClassifier:
    """
    Returns a fixed fill level with high confidence.
    Cycles through levels on repeated calls for visual variety during testing.
    """
    _cycle = ["medium", "high", "low", "full", "empty"]
    _idx   = 0

    def predict(self, image_rgb: np.ndarray) -> dict:
        label = self._cycle[StubFillClassifier._idx % len(self._cycle)]
        StubFillClassifier._idx += 1
        probs = {c: 0.02 for c in cfg.MODEL2_CLASSES}
        probs[label] = 0.88
        return {
            "label":      label,
            "confidence": 0.88,
            "probs":      probs
        }


# =============================================================================
# Real model wrappers
# =============================================================================

class FasterRCNNDetector:
    """
    Wraps the trained Faster R-CNN ResNet-50 FPN v2 for inference.
    Loaded lazily on first call to avoid slowing down Streamlit startup.
    """
    def __init__(self):
        self._model      = None
        self._label_map  = None


    def _load(self):
        import torch
        from torchvision.models.detection import (
            fasterrcnn_resnet50_fpn_v2,
            FasterRCNN_ResNet50_FPN_V2_Weights
        )
        from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


        if not cfg.MODEL1_PATH.exists():
            raise FileNotFoundError(
                f"Model #1 weights not found at {cfg.MODEL1_PATH}. "
                "Train the model first or set config.STUB_MODE = True."
            )

        with open(cfg.MODEL1_LABEL_MAP) as f:
            label_to_idx = json.load(f)

        self._idx_to_label = {v: k for k, v in label_to_idx.items()}
        self._idx_to_label[0] = "background"
        num_classes = len(label_to_idx) + 1

        model = fasterrcnn_resnet50_fpn_v2(weights = None)
        
        # get number of input features
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

        state = torch.load(cfg.MODEL1_PATH, map_location = "cpu")
        model.load_state_dict(state)
        model.eval()

        # Apply inference thresholds
        model.roi_heads.score_thresh = cfg.MODEL1_SCORE_THRESH
        model.roi_heads.nms_thresh   = cfg.MODEL1_NMS_THRESH

        self._model = model
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model.to(self._device)


    def predict(self, image_rgb: np.ndarray) -> list[dict]:
        """
        Run detection on an RGB image.

        Returns list of dicts, each with:
            box   : [x_min, y_min, x_max, y_max] pixels
            label : str class name
            score : float confidence
        """
        import torch
        import torchvision.transforms.functional as TF

        if self._model is None:
            self._load()

        tensor = TF.to_tensor(image_rgb).to(self._device)

        with torch.no_grad():
            preds = self._model([tensor])[0]

        results = []
        for box, label_idx, score in zip(
            preds["boxes"].cpu().tolist(),
            preds["labels"].cpu().tolist(),
            preds["scores"].cpu().tolist()
        ):
            label = self._idx_to_label.get(label_idx, "unknown")
            if label == "background":
                continue
            results.append({"box": box, "label": label, "score": score})

        return results



class EfficientNetFillClassifier:
    """
    Wraps the trained EfficientNet-B0 fill level classifier for inference.
    Loaded lazily on first call.
    """
    def __init__(self):
        self._model = None
        self._vocab = None


    def _load(self):
        import torch
        from fastai.vision.all import load_learner

        if not cfg.MODEL2_PATH.exists():
            raise FileNotFoundError(
                f"Model #2 weights not found at {cfg.MODEL2_PATH}. "
                "Train the model first or set config.STUB_MODE = True."
            )

        # fastai's load_learner loads the full exported learner (.pkl / .pth)
        self._learn = load_learner(cfg.MODEL2_PATH)
        self._vocab = list(self._learn.dls.vocab)
        self._model = self._learn


    def predict(self, image_rgb: np.ndarray) -> dict:
        """
        Classify fill level from a cropped container RGB image.

        Returns dict with:
            label      : str predicted fill level
            confidence : float (max softmax probability)
            probs      : dict {class: probability}
        """
        from PIL import Image as PILImage
        import torch

        if self._model is None:
            self._load()

        pil_img = PILImage.fromarray(image_rgb)
        _, pred_idx, probs = self._learn.predict(pil_img)

        label      = self._vocab[pred_idx]
        confidence = float(probs.max())
        prob_dict  = {cls: float(p) for cls, p in zip(self._vocab, probs)}

        return {
            "label":      label,
            "confidence": confidence,
            "probs":      prob_dict
        }


# =============================================================================
# Factory functions — use these in pipeline stages
# =============================================================================

def get_detector():
    """
    Return the appropriate detector based on STUB_MODE.
    """
    if cfg.STUB_MODE:
        return StubDetector()
    return FasterRCNNDetector()


def get_fill_classifier():
    """
    Return the appropriate fill classifier based on STUB_MODE.
    """
    if cfg.STUB_MODE:
        return StubFillClassifier()
    return EfficientNetFillClassifier()