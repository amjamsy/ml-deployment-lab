import math
import joblib
import numpy as np
from pathlib import Path
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parent
model = joblib.load(BASE_DIR / "model" / "model.pkl")

FEATURES = [
    "MedInc", "HouseAge", "AveRooms", "AveBedrms",
    "Population", "AveOccup", "Latitude", "Longitude"
]

@app.get("/health")
def health():
    return jsonify(ok=True)

@app.post("/predict")
def predict():
    data = request.get_json(silent=True)

    if not isinstance(data, dict):
        return jsonify(error="Invalid JSON"), 400

    values = data.get("features")

    if not isinstance(values, list) or len(values) != 8:
        return jsonify(error="Expected 8 features"), 400

    try:
        numbers = [float(value) for value in values]

        if not all(math.isfinite(value) for value in numbers):
            raise ValueError("Invalid number")

        X = np.array([numbers], dtype=float)
        prediction = float(model.predict(X)[0])

        return jsonify(prediction=prediction)

    except (ValueError, TypeError, OverflowError):
        return jsonify(error="Invalid features"), 400

@app.get("/")
def home():
    return render_template("main.html")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
