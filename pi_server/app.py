"""
INSALO demo server
-------------------
Runs ON THE RASPBERRY PI, next to the real neuralnetwork.py / Arduino code.
Wraps INSALOController in a tiny Flask API so a laptop on the same network
can send hypothetical CGM/context values and see exactly what the controller
would decide - without touching the motor.

This does NOT drive the pump. It's a read-only "what would you do" endpoint
for demoing the decision logic live. Wire in the real motor call yourself
wherever you already do it in your Arduino integration.

Run:
    pip install -r requirements.txt
    python app.py

Then find this Pi's local IP (e.g. `hostname -I` on the Pi) and enter it
into the laptop UI.
"""

import os
import sys

from flask import Flask, request, jsonify
from flask_cors import CORS

# neuralnetwork.py lives in insalo/scripts/, not next to this file —
# add it to the import path. Adjust this if you move pi_server/ elsewhere
# relative to the repo root.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

from neuralnetwork import (
    INSALOController,
    CYCLE_MULTIPLIERS,
    EXERCISE_MULTIPLIERS,
    STRESS_MULTIPLIERS,
    FEATURE_NAMES,
    encode,
    normalise,
)

app = Flask(__name__)
CORS(app)  # allow the laptop browser to call this across the LAN

# connect=False: the demo server never talks to the Arduino, so don't try
# to open the serial port (avoids a 2s startup delay / warning on machines
# with no Arduino attached, e.g. your laptop).
ctrl = INSALOController(connect=False)

WEIGHTS_FILE = os.path.join(os.path.dirname(__file__), "..", "scripts", "insalo_weights.json")

if os.path.exists(WEIGHTS_FILE):
    print("[BOOT] Loading pre-trained weights from {}...".format(WEIGHTS_FILE))
    ctrl.load_model(WEIGHTS_FILE)
else:
    print("[BOOT] No saved weights found — training now (this runs once)...")
    # Point csv_path at your real cleaned CSV if it's on this machine;
    # otherwise it silently falls back to synthetic training data.
    ctrl.train(csv_path="data/processed/cleaned_medtronic_data.csv")
    ctrl.save_model(WEIGHTS_FILE)
print("[BOOT] Ready.")


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "trained": ctrl.trained})


@app.route("/options", methods=["GET"])
def options():
    return jsonify({
        "exercise": list(EXERCISE_MULTIPLIERS.keys()),
        "stress": list(STRESS_MULTIPLIERS.keys()),
        "cycle_phase": list(CYCLE_MULTIPLIERS.keys()),
        "feature_names": FEATURE_NAMES,
    })


@app.route("/decide", methods=["POST"])
def decide():
    data = request.get_json(force=True) or {}
    try:
        bgl = float(data.get("bgl", 6.1))
        bgl_trend = float(data.get("bgl_trend", 0.0))
        exercise = data.get("exercise", "none")
        stress = data.get("stress", "low")
        cycle_phase = data.get("cycle_phase", "follicular")
        carbs_g = float(data.get("carbs_g", 0))
        hours_since_bolus = float(data.get("hours_since_bolus", 4.0))
        cgm_active = bool(data.get("cgm_active", True))

        result = ctrl.decide(
            bgl=bgl,
            bgl_trend=bgl_trend,
            exercise=exercise,
            stress=stress,
            cycle_phase=cycle_phase,
            carbs_g=carbs_g,
            hours_since_bolus=hours_since_bolus,
            cgm_active=cgm_active,
        )

        # Optional: also return per-layer activations so the UI can animate
        # the network "lighting up" for the demo. Purely cosmetic.
        activations_out = None
        if cgm_active and ctrl.trained:
            features = encode(bgl, bgl_trend, exercise, stress, cycle_phase,
                               carbs_g, hours_since_bolus)
            norm_f = normalise(features)
            activations, _ = ctrl.net.forward(norm_f)
            activations_out = [layer for layer in activations]

        return jsonify({"ok": True, "activations": activations_out, **result})

    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 400


if __name__ == "__main__":
    # 0.0.0.0 so the laptop/LAN can reach it, not just localhost.
    # Using 5050 instead of 5000 — on macOS, 5000 is often taken by
    # AirPlay Receiver (System Settings > General > AirDrop & Handoff).
    app.run(host="0.0.0.0", port=5050)