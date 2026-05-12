"""
TerraFlow BioAI — Flask API (Combined Model)
=============================================
Endpoints:
  GET  /           → API info
  GET  /health     → health check
  POST /predict    → irrigation prediction (RF model)
  POST /recommend  → crop recommendations (scorer)
  POST /full       → both prediction + recommendations in one call
"""

from flask import Flask, request, jsonify
import joblib
import numpy as np
import os

app = Flask(__name__)

# ── Load combined model once at startup ────────────────────
MODEL_PATH = os.path.join(os.path.dirname(__file__), "terraflow_combined_model.pkl")
pkg        = joblib.load(MODEL_PATH)
rf_model   = pkg["rf_model"]
RF_FEATURES    = pkg["rf_features"]
CROP_PROFILES  = pkg["crop_profiles"]
print(f"✓ TerraFlow combined model loaded  (v{pkg['version']})")


# ─────────────────────────────────────────────────────────────
# INDEX HELPERS
# ─────────────────────────────────────────────────────────────

def compute_vpd(ambient_temp, humidity):
    svp = 0.6108 * np.exp((17.27 * ambient_temp) / (ambient_temp + 237.3))
    avp = svp * (humidity / 100)
    return float(np.clip(svp - avp, 0, 8))


def compute_wsi(moisture, soil_temp, lux, humidity, ambient_temp):
    moisture_factor = (100 - moisture) / 100
    temp_factor     = np.clip((soil_temp - 20) / 20, 0, 1)
    lux_factor      = np.clip(lux / 80000, 0, 1)
    vpd_norm        = np.clip(compute_vpd(ambient_temp, humidity) / 4, 0, 1)
    wsi = (0.45 * moisture_factor +
           0.25 * temp_factor +
           0.15 * lux_factor +
           0.15 * vpd_norm)
    return float(np.clip(wsi, 0, 1))


# ─────────────────────────────────────────────────────────────
# PREDICTION LOGIC
# ─────────────────────────────────────────────────────────────

def run_irrigation_predict(sensor: dict) -> dict:
    vpd = sensor.get("vpd", compute_vpd(sensor["ambient_temp"], sensor["humidity"]))
    wsi = sensor.get("wsi", compute_wsi(
        sensor["moisture"], sensor["soil_temp"], sensor["lux"],
        sensor["humidity"], sensor["ambient_temp"]))

    X = np.array([[
        sensor["moisture"], sensor["soil_temp"], sensor["tds"],
        sensor["ambient_temp"], sensor["humidity"], sensor["pressure"],
        sensor["lux"], vpd, wsi
    ]])

    label      = int(rf_model.predict(X)[0])
    confidence = float(rf_model.predict_proba(X)[0][label])

    return {
        "prediction": "IRRIGATE" if label == 1 else "NO_IRRIGATE",
        "confidence": round(confidence, 4),
        "label":      label,
        "indices": {
            "vpd": round(float(vpd), 4),
            "wsi": round(float(wsi), 4),
        }
    }


def _score_param(value, low, high):
    if low <= value <= high:
        return 1.0
    elif value < low:
        return max(0.0, 1.0 - (low - value) / (low + 1e-6))
    else:
        return max(0.0, 1.0 - (value - high) / (high + 1e-6))


def run_crop_recommend(sensor: dict) -> list:
    results = []
    for crop, profile in CROP_PROFILES.items():
        scores = {
            "moisture":  _score_param(sensor["moisture"],  *profile["moisture"]),
            "soil_temp": _score_param(sensor["soil_temp"], *profile["soil_temp"]),
            "humidity":  _score_param(sensor["humidity"],  *profile["humidity"]),
            "tds":       _score_param(sensor["tds"],       *profile["tds"]),
            "lux":       _score_param(sensor["lux"],       *profile["lux"]),
        }
        total = round((0.30 * scores["moisture"] +
                       0.25 * scores["soil_temp"] +
                       0.20 * scores["humidity"] +
                       0.15 * scores["tds"] +
                       0.10 * scores["lux"]) * 100, 1)

        if total >= 80:   grade = "Excellent"
        elif total >= 60: grade = "Good"
        elif total >= 40: grade = "Moderate"
        else:             grade = "Poor"

        results.append({
            "crop":        crop,
            "score":       total,
            "grade":       grade,
            "description": profile["description"],
            "breakdown":   {k: round(v * 100, 1) for k, v in scores.items()}
        })

    results.sort(key=lambda x: x["score"], reverse=True)
    return results


# ─────────────────────────────────────────────────────────────
# VALIDATION HELPER
# ─────────────────────────────────────────────────────────────

REQUIRED_FIELDS = ["moisture", "soil_temp", "tds", "ambient_temp",
                   "humidity", "pressure", "lux"]

def validate_and_parse(data):
    """Returns (sensor_dict, error_string)"""
    if not data:
        return None, "Request body must be JSON"
    missing = [f for f in REQUIRED_FIELDS if f not in data]
    if missing:
        return None, f"Missing fields: {missing}"
    try:
        sensor = {f: float(data[f]) for f in REQUIRED_FIELDS}
        if "vpd" in data: sensor["vpd"] = float(data["vpd"])
        if "wsi" in data: sensor["wsi"] = float(data["wsi"])
        return sensor, None
    except ValueError as e:
        return None, f"Invalid value: {e}"


# ─────────────────────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────────────────────

@app.route("/", methods=["GET"])
def index():
    return jsonify({
        "service":  "TerraFlow BioAI Prediction API",
        "version":  pkg["version"],
        "endpoints": {
            "POST /predict":   "Irrigation prediction (IRRIGATE / NO_IRRIGATE + confidence)",
            "POST /recommend": "Crop recommendations (8 crops ranked by score)",
            "POST /full":      "Both prediction + recommendations in one call",
            "GET  /health":    "Health check"
        },
        "required_fields": REQUIRED_FIELDS,
        "optional_fields":  ["vpd", "wsi"]
    })


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "model_loaded": True, "version": pkg["version"]}), 200


@app.route("/predict", methods=["POST"])
def predict():
    sensor, err = validate_and_parse(request.get_json(silent=True))
    if err:
        return jsonify({"error": err}), 400
    try:
        result = run_irrigation_predict(sensor)
        result["input"] = sensor
        return jsonify(result), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/recommend", methods=["POST"])
def recommend():
    sensor, err = validate_and_parse(request.get_json(silent=True))
    if err:
        return jsonify({"error": err}), 400
    try:
        crops = run_crop_recommend(sensor)
        return jsonify({
            "top_crop":      crops[0]["crop"],
            "top_score":     crops[0]["score"],
            "recommendations": crops,
            "input":         sensor
        }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/full", methods=["POST"])
def full():
    sensor, err = validate_and_parse(request.get_json(silent=True))
    if err:
        return jsonify({"error": err}), 400
    try:
        irrigation = run_irrigation_predict(sensor)
        crops      = run_crop_recommend(sensor)
        return jsonify({
            "irrigation":      irrigation,
            "top_crop":        crops[0]["crop"],
            "top_score":       crops[0]["score"],
            "recommendations": crops,
            "input":           sensor
        }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ─────────────────────────────────────────────────────────────
# RUN
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
