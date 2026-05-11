"""
TerraFlow BioAI — Flask Prediction API
=======================================
POST /predict   → irrigation prediction
GET  /health    → service health check
GET  /          → API info
"""

from flask import Flask, request, jsonify
import joblib
import numpy as np
import os

app = Flask(__name__)

# ── Load model once at startup ──────────────────────────────
MODEL_PATH = os.path.join(os.path.dirname(__file__), "terraflow_rf_model.joblib")
model = joblib.load(MODEL_PATH)
print(f"✓ Model loaded from {MODEL_PATH}")

FEATURES = ["moisture", "soil_temp", "tds", "ambient_temp",
            "humidity", "pressure", "lux", "wsi", "dri"]


# ── Index helpers ───────────────────────────────────────────
def compute_wsi(moisture, soil_temp, lux, humidity):
    moisture_factor = (100 - moisture) / 100
    temp_factor     = np.clip((soil_temp - 20) / 20, 0, 1)
    lux_factor      = np.clip(lux / 80000, 0, 1)
    vpd_approx      = (1 - humidity / 100) * 0.6
    wsi = 0.45 * moisture_factor + 0.25 * temp_factor + 0.15 * lux_factor + 0.15 * vpd_approx
    return float(np.clip(wsi, 0, 1))


def compute_dri(humidity, ambient_temp, pressure):
    hum_factor   = np.clip((humidity - 50) / 50, 0, 1)
    temp_factor  = np.clip(1 - abs(ambient_temp - 25) / 15, 0, 1)
    press_factor = np.clip((1013 - pressure) / 30, 0, 1)
    dri = 0.5 * hum_factor + 0.3 * temp_factor + 0.2 * press_factor
    return float(np.clip(dri, 0, 1))


# ── Routes ──────────────────────────────────────────────────

@app.route("/", methods=["GET"])
def index():
    return jsonify({
        "service": "TerraFlow BioAI Prediction API",
        "version": "1.0.0",
        "endpoints": {
            "POST /predict": "Run irrigation prediction",
            "GET  /health":  "Health check"
        },
        "required_fields": ["moisture", "soil_temp", "tds",
                             "ambient_temp", "humidity", "pressure", "lux"],
        "optional_fields": ["wsi", "dri"]
    })


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "model_loaded": model is not None}), 200


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True)

    if not data:
        return jsonify({"error": "Request body must be JSON"}), 400

    # ── Validate required fields ────────────────────────────
    required = ["moisture", "soil_temp", "tds", "ambient_temp",
                "humidity", "pressure", "lux"]
    missing = [f for f in required if f not in data]
    if missing:
        return jsonify({"error": f"Missing fields: {missing}"}), 400

    try:
        # ── Extract values ──────────────────────────────────
        moisture     = float(data["moisture"])
        soil_temp    = float(data["soil_temp"])
        tds          = float(data["tds"])
        ambient_temp = float(data["ambient_temp"])
        humidity     = float(data["humidity"])
        pressure     = float(data["pressure"])
        lux          = float(data["lux"])

        # ── Compute or accept indices ───────────────────────
        wsi = float(data["wsi"]) if "wsi" in data else compute_wsi(moisture, soil_temp, lux, humidity)
        dri = float(data["dri"]) if "dri" in data else compute_dri(humidity, ambient_temp, pressure)

        # ── Run inference ───────────────────────────────────
        X = np.array([[moisture, soil_temp, tds, ambient_temp,
                        humidity, pressure, lux, wsi, dri]])

        label      = int(model.predict(X)[0])
        proba      = model.predict_proba(X)[0]
        confidence = float(proba[label])

        return jsonify({
            "prediction":  "IRRIGATE" if label == 1 else "NO_IRRIGATE",
            "confidence":  round(confidence, 4),
            "label":       label,
            "indices": {
                "wsi": round(wsi, 4),
                "dri": round(dri, 4)
            },
            "input_received": {
                "moisture": moisture, "soil_temp": soil_temp,
                "tds": tds, "ambient_temp": ambient_temp,
                "humidity": humidity, "pressure": pressure, "lux": lux
            }
        }), 200

    except ValueError as e:
        return jsonify({"error": f"Invalid value: {str(e)}"}), 400
    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500


# ── Run ─────────────────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
