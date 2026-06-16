from __future__ import annotations

import io
import os
import threading
from pathlib import Path

import numpy as np
import torch
from torchvision import models
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from flask import Flask, jsonify, request, send_from_directory
from PIL import Image, ImageOps


CLASS_NAMES = {
    0: "Lettuce",
    1: "Potato",
    2: "Carrot",
    3: "Onion",
    4: "Garlic",
    5: "Leek",
    6: "Broccoli",
}


def resolve_device():
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def get_model(num_model_classes, pretrained=False):
    if pretrained:
        weights = models.detection.FasterRCNN_ResNet50_FPN_Weights.DEFAULT
        model = models.detection.fasterrcnn_resnet50_fpn(weights=weights)
    else:
        model = models.detection.fasterrcnn_resnet50_fpn(weights=None, weights_backbone=None)

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_model_classes)
    return model


NUM_MODEL_CLASSES = len(CLASS_NAMES) + 1
DEVICE = resolve_device()
UPLOAD_DIR = Path(os.environ.get("UPLOAD_DIR", "/tmp/vegetable-web"))
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
DEFAULT_MODEL_PATH = os.environ.get("MODEL_PATH", "./best_model.pth")
SCORE_THRESHOLD = float(os.environ.get("SCORE_THRESHOLD", 0.5))

WEB_DIR = Path(__file__).resolve().parent
app = Flask(__name__, static_folder=str(WEB_DIR), static_url_path="")
_model_lock = threading.Lock()
_model = None
_model_source = None


def load_model_from_path(model_path: str):
    resolved_path = Path(model_path)
    if not resolved_path.exists():
        raise FileNotFoundError(f"Model file not found: {resolved_path}")

    model = get_model(NUM_MODEL_CLASSES, pretrained=False)
    state_dict = torch.load(resolved_path, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()
    return model


def set_loaded_model(model, source: str):
    global _model, _model_source
    with _model_lock:
        _model = model
        _model_source = source


def get_loaded_model():
    with _model_lock:
        return _model, _model_source


if DEFAULT_MODEL_PATH:
    try:
        set_loaded_model(load_model_from_path(DEFAULT_MODEL_PATH), DEFAULT_MODEL_PATH)
    except Exception:
        pass


@app.get("/")
def index():
    return send_from_directory(WEB_DIR, "index.html")


@app.get("/health")
def health():
    model, source = get_loaded_model()
    return jsonify(
        {
            "ok": True,
            "device": str(DEVICE),
            "model_loaded": model is not None,
            "model_source": source,
        }
    )


@app.post("/load-model")
def load_model_route():
    uploaded_file = request.files.get("model")
    model_path = None
    if request.is_json:
        payload = request.get_json(silent=True) or {}
        model_path = payload.get("model_path")
    else:
        model_path = request.form.get("model_path")

    try:
        if uploaded_file and uploaded_file.filename:
            target_path = UPLOAD_DIR / uploaded_file.filename
            uploaded_file.save(target_path)
            loaded_model = load_model_from_path(str(target_path))
            set_loaded_model(loaded_model, str(target_path))
            return jsonify({"ok": True, "message": "Model loaded", "source": str(target_path)})

        if model_path:
            loaded_model = load_model_from_path(model_path)
            set_loaded_model(loaded_model, model_path)
            return jsonify({"ok": True, "message": "Model loaded", "source": model_path})

        return jsonify({"ok": False, "message": "Provide a model file or model_path."}), 400
    except Exception as exc:
        return jsonify({"ok": False, "message": str(exc)}), 400


@app.post("/predict")
def predict():
    model, _ = get_loaded_model()
    if model is None:
        return jsonify({"ok": False, "message": "No model loaded."}), 400

    image_file = request.files.get("image")
    if image_file is None:
        return jsonify({"ok": False, "message": "No image uploaded."}), 400

    image = ImageOps.exif_transpose(Image.open(io.BytesIO(image_file.read()))).convert("RGB")
    image_array = np.array(image, dtype=np.float32) / 255.0
    image_tensor = torch.from_numpy(image_array).permute(2, 0, 1)

    with torch.no_grad():
        output = model([image_tensor.to(DEVICE)])[0]

    width, height = image.size
    predictions = []
    boxes = output.get("boxes", torch.empty((0, 4))).detach().cpu().tolist()
    labels = output.get("labels", torch.empty((0,), dtype=torch.int64)).detach().cpu().tolist()
    scores = output.get("scores", torch.empty((0,))).detach().cpu().tolist()

    for box, label, score in zip(boxes, labels, scores):
        if score < SCORE_THRESHOLD:
            continue
        class_name = CLASS_NAMES.get(int(label) - 1, f"ID: {int(label) - 1}")
        predictions.append(
            {
                "label": class_name,
                "score": round(float(score), 4),
                "box": [round(float(v), 2) for v in box],
            }
        )

    return jsonify(
        {
            "ok": True,
            "width": width,
            "height": height,
            "predictions": predictions,
        }
    )


if __name__ == "__main__":
    print("Web app started. Open http://localhost:8000")
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8000)), debug=False)
