"""
utils.py — Data loading, model initialisation, and prediction helpers.

Keeps all ML-related logic out of app.py so the Flask routes stay clean.
"""

import os
import numpy as np
import pandas as pd
import torch
import torchvision.transforms.functional as TF
from PIL import Image

from model import CNN

# ── Paths ────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "plant_disease_model_1_latest.pt")
DISEASE_CSV = os.path.join(BASE_DIR, "disease_info.csv")
SUPPLEMENT_CSV = os.path.join(BASE_DIR, "supplement_info.csv")
UPLOAD_DIR = os.path.join(BASE_DIR, "static", "uploads")

# ── Data ─────────────────────────────────────────────────────────────
disease_info = pd.read_csv(DISEASE_CSV, encoding="cp1252")
supplement_info = pd.read_csv(SUPPLEMENT_CSV, encoding="cp1252")

# ── Model ────────────────────────────────────────────────────────────
model = CNN(num_classes=39)
model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device("cpu")))
model.eval()


# ── Prediction helpers ───────────────────────────────────────────────
def _preprocess(image_path: str) -> torch.Tensor:
    """Load an image, resize to 224×224, and convert to a batch tensor."""
    image = Image.open(image_path).resize((224, 224))
    tensor = TF.to_tensor(image)
    return tensor.view(-1, 3, 224, 224)


def predict(image_path: str) -> int:
    """Return the predicted class index for the given image."""
    input_data = _preprocess(image_path)
    output = model(input_data)
    return int(np.argmax(output.detach().numpy()))


def predict_with_confidence(image_path: str) -> tuple[int, float]:
    """Return (class_index, confidence_percentage) for the given image."""
    input_data = _preprocess(image_path)
    output = model(input_data)
    probabilities = torch.nn.functional.softmax(output, dim=1)
    confidence, predicted_class = torch.max(probabilities, dim=1)
    return predicted_class.item(), round(confidence.item() * 100, 2)


def get_disease_info(index: int) -> dict:
    """Return a dict with all disease + supplement details for a class index."""
    return {
        "disease_name":     str(disease_info["disease_name"][index]),
        "description":      str(disease_info["description"][index]),
        "possible_steps":   str(disease_info["Possible Steps"][index]),
        "disease_image_url": str(disease_info["image_url"][index]),
        "supplement": {
            "name":      str(supplement_info["supplement name"][index]),
            "image_url": str(supplement_info["supplement image"][index]),
            "buy_link":  str(supplement_info["buy link"][index]),
        },
    }


def allowed_file(filename: str) -> bool:
    """Check whether the uploaded file has an allowed image extension."""
    ALLOWED = {"png", "jpg", "jpeg", "webp", "bmp"}
    return "." in filename and filename.rsplit(".", 1)[-1].lower() in ALLOWED


def ensure_upload_dir() -> None:
    """Create the upload directory if it doesn't exist."""
    os.makedirs(UPLOAD_DIR, exist_ok=True)
