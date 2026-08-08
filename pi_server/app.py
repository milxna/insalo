"""
INSALO Flask demo server
"""

import os
import sys

from flask import Flask, request, jsonify
from flask_cors import CORS

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

from neuralnetwork import (
    INSALOController,
    kCycleMultipliers,
    kExerciseMultipliers,
    kStressMultipliers,
    FEATURE_NAMES,
    kCGMValidMin,
    kCGMValidMax,
    kMaxLimit,
    kTargetBGL,
    kCarbRatio,
    kMaxCarbBolus,
    encode,
    normalise,
)

app = Flask(__name__)
CORS(app)

ctrl = INSALOController(connect=True)

WEIGHTS_FILE = os.path.join(
    os.path.dirname(__file__), "..", "scripts", "insalo_weights.json"
)

if os.path.exists(WEIGHTS_FILE):
    print("[BOOT] Loading pre-trained weights from {}...".format(WEIGHTS_FILE))
    ctrl.load_model(WEIGHTS_FILE)
else:
    print("[BOOT] No saved weights found — training now (this runs once)...")
    ctrl.train(csv_path="data/processed/cleaned_medtronic_data.csv")
    ctrl.save_model(WEIGHTS_FILE)

print("[BOOT] Ready.")


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "trained": ctrl.trained})


@app.route("/options", methods=["GET"])
def options():
    return jsonify({
        "exercise": list(kExerciseMultipliers.keys()),
        "stress": list(kStressMultipliers.keys()),
        "cycle_phase": list(kCycleMultipliers.keys()),
        "feature_names": FEATURE_NAMES,
        "max_delivery": kMaxLimit,
        "cgm_valid_min": kCGMValidMin,
        "cgm_valid_max": kCGMValidMax,
        "target_bgl": kTargetBGL,
        "insulin_to_carb_ratio": kCarbRatio,
        "max_carb_bolus": kMaxCarbBolus,
    })


@app.route("/decide", methods=["POST"])
def decide():
    """
    Continuous 5-minute correction loop only.
    Carb intake is handled separately by /carb-bolus.
    """
    data = request.get_json(force=True) or {}

    try:
        bgl = float(data.get("bgl", 6.1))
        bgl_trend = float(data.get("bgl_trend", 0.0))
        exercise = data.get("exercise", "none")
        stress = data.get("stress", "low")
        cycle_phase = data.get("cycle_phase", "follicular")
        hours_since_bolus = float(data.get("hours_since_bolus", 4.0))
        cgm_active = bool(data.get("cgm_active", True))

        result = ctrl.decide(
            bgl=bgl,
            bgl_trend=bgl_trend,
            exercise=exercise,
            stress=stress,
            cycle_phase=cycle_phase,
            hours_since_bolus=hours_since_bolus,
            cgm_active=cgm_active,
        )

        # Used only to expose the NN activations in the demo UI.
        activations_out = None
        is_auto = (
            cgm_active
            and ctrl.trained
            and kCGMValidMin < bgl <= kCGMValidMax
        )

        if is_auto:
            features = encode(
                bgl,
                bgl_trend,
                exercise,
                stress,
                cycle_phase,
                hours_since_bolus,
            )
            norm_f = normalise(features)
            activations, _ = ctrl.net.forward(norm_f)
            activations_out = [layer for layer in activations]

        return jsonify({
            "ok": True,
            "activations": activations_out,
            **result,
        })

    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 400


@app.route("/carb-bolus", methods=["POST"])
def carb_bolus():
    """
    Carb bolus, separate from the continuous basal/correction loop.
    """
    data = request.get_json(force=True) or {}

    try:
        carbs_g = float(data.get("carbs_g", 0))
        ratio = float(data.get("insulin_to_carb_ratio", kCarbRatio))

        result = ctrl.carbBolus(carbs_g, carbRatio=ratio)

        return jsonify({"ok": True, **result})

    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 400


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5050)