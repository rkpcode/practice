import os
import logging
from typing import List
from pathlib import Path
import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify

"""
Simple TensorFlow model deployment example as a minimal Flask app.

- POST /predict with JSON: {"instances": [[...], [...], ...]}
- Responds with JSON: {"predictions": [[...], ...]}

Usage:
    pip install tensorflow flask numpy
    python main.py
"""



try:
except Exception as e:
        raise ImportError("TensorFlow is required. Install with `pip install tensorflow`.") from e

try:
except Exception as e:
        raise ImportError("Flask is required. Install with `pip install flask`.") from e


LOG = logging.getLogger("tf-deploy")
logging.basicConfig(level=logging.INFO)

MODEL_DIR = Path(__file__).with_suffix("") / "model"  # creates 'main/model' next to this file
MODEL_DIR.mkdir(parents=True, exist_ok=True)
MODEL_PATH = str(MODEL_DIR / "simple_dense")


def create_and_save_model(path: str):
        """Create a tiny Keras model and save it to path."""
        LOG.info("Creating new model at %s", path)
        model = tf.keras.Sequential(
                [
                        tf.keras.layers.Input(shape=(4,)),  # example input shape: 4 features
                        tf.keras.layers.Dense(16, activation="relu"),
                        tf.keras.layers.Dense(1, activation="linear"),
                ],
                name="tiny_dense",
        )
        model.compile(optimizer="adam", loss="mse")
        # save immediately (weights are random but deterministic enough for demo)
        model.save(path)
        return model


def load_or_create_model(path: str):
        if os.path.exists(path):
                LOG.info("Loading model from %s", path)
                return tf.keras.models.load_model(path)
        return create_and_save_model(path)


app = Flask(__name__)
model = load_or_create_model(MODEL_PATH)


def validate_instances(data) -> np.ndarray:
        if not isinstance(data, list):
                raise ValueError("JSON body must contain a top-level 'instances' list of feature lists.")
        arr = np.array(data, dtype=float)
        if arr.ndim == 1:
                arr = arr.reshape(1, -1)
        return arr


@app.route("/predict", methods=["POST"])
def predict():
        if not request.is_json:
                return jsonify({"error": "Request must be application/json"}), 400
        payload = request.get_json()
        if "instances" not in payload:
                return jsonify({"error": "JSON must contain 'instances' key"}), 400
        try:
                instances = validate_instances(payload["instances"])
        except ValueError as e:
                return jsonify({"error": str(e)}), 400
        try:
                preds = model.predict(instances)
        except Exception as e:
                LOG.exception("Prediction failed")
                return jsonify({"error": "Model prediction failed", "details": str(e)}), 500
        # Convert to native Python types
        predictions = preds.tolist()
        return jsonify({"predictions": predictions})


if __name__ == "__main__":
        # Example: run with `python main.py` and POST JSON to http://127.0.0.1:5000/predict
        LOG.info("Starting Flask server with model at %s", MODEL_PATH)
        app.run(host="0.0.0.0", port=5000, debug=False)