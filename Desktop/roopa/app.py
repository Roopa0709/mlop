import math
from pathlib import Path

import mlflow
import pandas as pd
from flask import Flask, jsonify, request


PROJECT_DIR = Path(__file__).resolve().parent
TRACKING_DB = PROJECT_DIR / "mlflow.db"
MODEL_URI = "models:/california-housing-price-predictor@champion"
FEATURE_NAMES = (
    "MedInc",
    "HouseAge",
    "AveRooms",
    "AveBedrms",
    "Population",
    "AveOccup",
    "Latitude",
    "Longitude",
)


def load_model():
    mlflow.set_tracking_uri(f"sqlite:///{TRACKING_DB.as_posix()}")
    return mlflow.pyfunc.load_model(MODEL_URI)


def create_app(model=None):
    app = Flask(__name__)
    app.config["MODEL"] = model if model is not None else load_model()

    @app.get("/health")
    def health():
        return jsonify({"status": "ok"})

    @app.route("/predict", methods=["GET", "POST"])
    def predict():
        if request.method == "GET":
            features = request.args.to_dict()
        else:
            if not request.is_json:
                return jsonify({"error": "Content-Type must be application/json"}), 400

            payload = request.get_json(silent=True)
            if not isinstance(payload, dict) or not isinstance(
                payload.get("features"), dict
            ):
                return jsonify(
                    {"error": "Request body must contain a 'features' object"}
                ), 400
            features = payload["features"]

        missing = sorted(set(FEATURE_NAMES) - set(features))
        unexpected = sorted(set(features) - set(FEATURE_NAMES))
        if missing or unexpected:
            return jsonify(
                {
                    "error": "Feature names do not match the model input",
                    "missing": missing,
                    "unexpected": unexpected,
                }
            ), 400

        values = {}
        for name in FEATURE_NAMES:
            value = features[name]
            if request.method == "GET":
                try:
                    value = float(value)
                except ValueError:
                    return jsonify({"error": f"Feature '{name}' must be a number"}), 400
            elif isinstance(value, bool) or not isinstance(value, (int, float)):
                return jsonify({"error": f"Feature '{name}' must be a number"}), 400
            if not math.isfinite(value):
                return jsonify({"error": f"Feature '{name}' must be finite"}), 400
            values[name] = float(value)

        input_frame = pd.DataFrame([values], columns=FEATURE_NAMES)
        prediction = float(app.config["MODEL"].predict(input_frame)[0])
        return jsonify(
            {
                "prediction": prediction,
                "unit": "$100,000s",
                "estimated_price_usd": prediction * 100000,
            }
        )

    return app


app = create_app()


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8000, debug=False)
