import math
import random
import csv
import os
import serial
import json
import time

# CONSTANTS
TARGET_BGL    = 6.1    # mmol/L
SAFE_DELIVERY    = 1.5    # U/h    - AS PER CRITERION 2
MAX_DELIVERY     = 6.0    # U/h  
BGL_LOW_REF, BGL_LOW_REF_DOSE   = 8.0, 0.20   # mmol/L, U/cycle
BGL_SATURATION_TARGET           = 16.5        # mmol/L - dose saturates at MAX_DELIVERY here

_rate_low  = BGL_LOW_REF_DOSE / (5.0 / 60.0)
_rate_high = MAX_DELIVERY
BGL_CORRECTION_GAIN = (_rate_high - _rate_low) / (BGL_SATURATION_TARGET - BGL_LOW_REF)
DEFAULT_DELIVERY    = _rate_low - BGL_CORRECTION_GAIN * (BGL_LOW_REF - TARGET_BGL)

BGL_SATURATION_TARGET = 16.5   # mmol/L - highest BGL we still want a distinct dose for
BGL_CORRECTION_GAIN   = (MAX_DELIVERY - DEFAULT_DELIVERY) / (BGL_SATURATION_TARGET - TARGET_BGL)
MIN_DELIVERY     = 0.0    # U/h
DOSE_INCREMENT   = 0.05 
CGM_VALID_MIN = 4.0    # mmol/L - below this, glucose is genuinely low: SUSPEND, don't dose
CGM_VALID_MAX = 22.0   # mmol/L - above this, reading is implausible: treat as sensor error

# MOTOR CALIBRATION
CONTROL_PERIOD_MIN = 5.0 / 60.0      # run every 5 mins
STEPS_PER_REV = 200
LEADSCREW_PITCH = 1.25
MM_PER_UNIT = 0.32         # fixme - calibrate
STEPS_PER_MM = STEPS_PER_REV / LEADSCREW_PITCH
STEPS_PER_UNIT = MM_PER_UNIT * STEPS_PER_MM

# PHYSIOLOGICAL FACTORS
CYCLE_MULTIPLIERS = {
    "follicular": 1.0,
    "ovulation":  0.95,
    "luteal":     1.2,
    "menstrual":  0.9,
}

EXERCISE_MULTIPLIERS = {
    "none":     1.0,
    "light":    0.9,    # walking, yoga (10% decrease in insulin)
    "moderate": 0.8,    # swimming, cycling (20% decrease in insulin)
    "high":     0.65,   # AFL, HIIT, contact sport (35% decrease in insulin)
}

STRESS_MULTIPLIERS = {
    "low":    1.00,  # standard insulin delivery
    "medium": 1.05,  # 5% increase in insulin
    "high":   1.1,   # 10% increase in insulin need
}


# FEATURE ENCODING
# fixme - implement a proper carb-bolus function, separate from decide()'s
# continuous correction, before carbs_g is used for anything but testing
# with carbs_g=0.
def encode(bgl, bglTrend, exercise, stress, cyclePhase,
           carbs_g=0, hoursSinceBolus=4.0):

    bgl_error  = bgl - TARGET_BGL
    exercise_f = EXERCISE_MULTIPLIERS.get(exercise, 1.0)
    stress_f   = STRESS_MULTIPLIERS.get(stress, 1.0)
    cycle_f    = CYCLE_MULTIPLIERS.get(cyclePhase, 1.0)
    iob        = max(0.0, 1.0 - hoursSinceBolus / 4.0)

    return [bgl, bgl_error, bglTrend, exercise_f, stress_f,
            cycle_f, carbs_g, iob]

FEATURE_NAMES = [
    "BGL (mmol/L)",
    "BGL error (from target)",
    "BGL trend (per 15 min)",
    "Exercise factor",
    "Stress factor",
    "Cycle phase factor",
    "Carbs (g) - PLACEHOLDER, see WARNING above encode()",
    "Insulin on board",
]

N_FEATURES = len(FEATURE_NAMES)


# NORMALISATION
featureMins = [2.0,   -3.0,  -3.0,  0.65,  1.00,  0.90,  0.0,   0.0]
featureMaxs = [25.0,  12.0,   3.0,  1.00,  1.30,  1.20,  60.0,  1.0]

def normalise(features):
    result = []
    for val, lo, hi in zip(features, featureMins, featureMaxs):
        span = hi - lo
        result.append((val - lo) / span if span > 0 else 0.0)
    return result

def normaliseTarget(delivery):
    return (delivery - MIN_DELIVERY) / (MAX_DELIVERY - MIN_DELIVERY)

def denormaliseTarget(norm_delivery):
    return norm_delivery * (MAX_DELIVERY - MIN_DELIVERY) + MIN_DELIVERY

def roundToIncrement(units, increment=DOSE_INCREMENT):
    """Rounds a dose to the nearest multiple of `increment` (default 0.05U).
    This is what actually gets converted to motor steps - it guarantees the
    dose you report is the dose you deliver, rather than whatever value
    falls out of rounding the motor's step count."""
    return round(units / increment) * increment

def deliveryToMicrobolus(deliveryRate):
    """Converts a U/h delivery rate into the exact number of insulin units
    (U) that will actually be delivered THIS control period (5 min).
    This is the real dose - deliveryRate on its own is only an hourly
    equivalent and is never injected as-is."""
    return deliveryRate * CONTROL_PERIOD_MIN

def microbolusToSteps(microbolus_units):
    """Converts an exact insulin-unit dose into motor steps."""
    return round(microbolus_units * STEPS_PER_UNIT)

def deliveryToSteps(deliveryRate):
    """Convenience wrapper: U/h rate -> motor steps for one control period."""
    return microbolusToSteps(deliveryToMicrobolus(deliveryRate))


# ACTIVATION FUNCTIONS

def relu(x):
    return max(0.0, x)

def reluDerivative(x):
    return 1.0 if x > 0 else 0.0

def sigmoid(x):
    x = max(-500.0, min(500.0, x))
    return 1.0 / (1.0 + math.exp(-x))

def sigmoidDerivative(sig_x):
    return sig_x * (1.0 - sig_x)


# THE NEURAL NETWORK
class NeuralNetwork:

    def __init__(self, layerSizes=(N_FEATURES, 16, 8, 1), learningRate=0.01):
        self.lr = learningRate
        self.layer_sizes = layerSizes
        self.weights = []
        self.biases  = []

        for i in range(len(layerSizes) - 1):
            n_in  = layerSizes[i]
            n_out = layerSizes[i + 1]
            scale = math.sqrt(2.0 / n_in)
            layer_w = [[random.gauss(0, scale) for _ in range(n_in)]
                       for _ in range(n_out)]
            layer_b = [0.0] * n_out
            self.weights.append(layer_w)
            self.biases.append(layer_b)

    def forward(self, x):
        activations     = [x]
        pre_activations = []

        for i, (W, b) in enumerate(zip(self.weights, self.biases)):
            prev = activations[-1]
            z = [sum(W[j][k] * prev[k] for k in range(len(prev))) + b[j]
                 for j in range(len(W))]
            pre_activations.append(z)

            is_last = (i == len(self.weights) - 1)
            if is_last:
                a = [sigmoid(zi) for zi in z]
            else:
                a = [relu(zi) for zi in z]

            activations.append(a)

        return activations, pre_activations

    def predictOne(self, x):
        activations, _ = self.forward(x)
        return activations[-1][0]

    def predict(self, X):
        return [self.predictOne(x) for x in X]

    def trainOne(self, x, y_true):
        activations, pre_activations = self.forward(x)
        prediction = activations[-1][0]

        output_error = 2.0 * (prediction - y_true)
        sig_deriv    = sigmoidDerivative(prediction)
        deltas = [[output_error * sig_deriv]]

        for i in reversed(range(len(self.weights) - 1)):
            W_next   = self.weights[i + 1]
            d_next   = deltas[0]
            z_curr   = pre_activations[i]

            d_curr = []
            for j in range(len(z_curr)):
                upstream = sum(d_next[k] * W_next[k][j]
                               for k in range(len(d_next)))
                d_curr.append(upstream * reluDerivative(z_curr[j]))

            deltas.insert(0, d_curr)

        for i in range(len(self.weights)):
            prev_activations = activations[i]
            for j in range(len(self.weights[i])):
                for k in range(len(self.weights[i][j])):
                    self.weights[i][j][k] -= self.lr * deltas[i][j] * prev_activations[k]
                self.biases[i][j] -= self.lr * deltas[i][j]

        return (prediction - y_true) ** 2

    def fit(self, X, y, epochs=200, verbose=True):
        n = len(X)
        for epoch in range(epochs):
            indices = list(range(n))
            random.shuffle(indices)

            total_loss = 0.0
            for i in indices:
                loss = self.trainOne(X[i], y[i])
                total_loss += loss

            avg_loss = total_loss / n

            if verbose and (epoch + 1) % 50 == 0:
                print("  Epoch {:>3}/{}  |  avg loss: {:.6f}".format(
                    epoch + 1, epochs, avg_loss))

        return self

    def score_r2(self, X, y):
        preds  = self.predict(X)
        mean_y = sum(y) / len(y)
        ss_res = sum((p - t) ** 2 for p, t in zip(preds, y))
        ss_tot = sum((t - mean_y) ** 2 for t in y)
        return 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    def accuracy_within(self, X, y_norm, tolerance_units=0.5):
        preds   = self.predict(X)
        correct = 0
        for p, t in zip(preds, y_norm):
            p_real = denormaliseTarget(p)
            t_real = denormaliseTarget(t)
            if abs(p_real - t_real) <= tolerance_units:
                correct += 1
        return 100.0 * correct / len(y_norm)

    def save_weights(self, filepath="model_weights.json"):
        data = {
            "layer_sizes": self.layer_sizes,
            "weights": self.weights,
            "biases": self.biases,
        }
        with open(filepath, "w") as f:
            json.dump(data, f)
        print(f"[INFO] Weights successfully saved to {filepath}")

    def load_weights(self, filepath="model_weights.json"):
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Weight file not found: {filepath}")

        with open(filepath, "r") as f:
            data = json.load(f)

        self.layer_sizes = data["layer_sizes"]
        self.weights = data["weights"]
        self.biases = data["biases"]
        print(f"[INFO] Weights successfully loaded from {filepath}")


# SYNTHETIC DATA

def generate_training_data(n=10000, seed=42):
    random.seed(seed)
    phases    = list(CYCLE_MULTIPLIERS)
    exercises = list(EXERCISE_MULTIPLIERS)
    stresses  = list(STRESS_MULTIPLIERS)

    X, y = [], []
    for _ in range(n):
        bgl       = random.uniform(3.5, 18.0)
        trend     = random.uniform(-2.5, 2.5)
        exercise  = random.choice(exercises)
        stress    = random.choice(stresses)
        phase     = random.choice(phases)
        carbs     = random.choice([0, 0, 0, 15, 30, 45, 60])
        iob_hours = random.uniform(0, 5)

        features   = encode(bgl, trend, exercise, stress, phase, carbs, iob_hours)
        norm_feats = normalise(features)

        bgl_error  = bgl - TARGET_BGL
        cycle_f    = CYCLE_MULTIPLIERS[phase]
        stress_f   = STRESS_MULTIPLIERS[stress]
        exercise_f = EXERCISE_MULTIPLIERS[exercise]
        delivery = (DEFAULT_DELIVERY
                 + BGL_CORRECTION_GAIN * bgl_error
                 + 0.15 * trend
                 + (cycle_f  - 1.0) * 1.5
                 + (stress_f - 1.0) * 0.8
                 - (1.0 - exercise_f) * 1.2
                 + 0.01 * carbs)  # PLACEHOLDER - should be a separate carb bolus, see WARNING above encode()
        delivery = max(MIN_DELIVERY, min(delivery, MAX_DELIVERY))

        X.append(norm_feats)
        y.append(normaliseTarget(delivery))

    return X, y


def load_cgm_csv(filepath):
    """Load CGM readings from Medtronic CSV export."""
    if not os.path.exists(filepath):
        print("[WARN] CSV not found at: {}".format(filepath))
        return []

    rows = []
    with open(filepath, newline='', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for row in reader:
            bgl_raw = (row.get('SG') or row.get('SGValue') or row.get('SG Value') or
           row.get('sg_value') or row.get('BGL') or
           row.get('Sensor Glucose (mmol/L)') or '')
            try:
                bgl = float(bgl_raw)
            except (ValueError, TypeError):
                continue
            rows.append({
                'bgl':       bgl,
                'timestamp': row.get('Timestamp', row.get('Date/Time', '')),
            })

    for i, row in enumerate(rows):
        row['bgl_trend'] = rows[i]['bgl'] - rows[i-3]['bgl'] if i >= 3 else 0.0

    print("[INFO] Loaded {} CGM readings from {}".format(
        len(rows), os.path.basename(filepath)))
    return rows


# CONTROLLER

class INSALOController:
    def __init__(self, arduino_port="/dev/ttyACM0", baudrate=115200, connect=True):
        self.serial = None
        if connect:
            try:
                self.serial = serial.Serial(arduino_port, baudrate, timeout=2)
                time.sleep(2)  # let the Arduino reset after the serial connection opens
                print(f"[INFO] Connected to Arduino on {arduino_port}")
            except serial.SerialException as e:
                print(f"[WARN] Could not connect to Arduino: {e}")
                self.serial = None

        self.net = NeuralNetwork(
            layerSizes=(N_FEATURES, 16, 8, 1), learningRate=0.01
        )
        self.trained = False

    def save_model(self, filepath="insalo_weights.json"):
        if not self.trained:
            print("not trained")
        self.net.save_weights(filepath)

    def load_model(self, filepath="insalo_weights.json"):
        self.net.load_weights(filepath)
        self.trained = True

    def train(self, csv_path=None, n_synthetic=2000, epochs=300,
              test_split=0.2, seed=42):

        random.seed(seed)
        if csv_path and os.path.exists(csv_path):
            cgm_rows = load_cgm_csv(csv_path)
            csv_path2 = os.path.join("data", "processed", "cleaned_medtronic_data2.csv")
            if os.path.exists(csv_path2):
                cgm_rows2 = load_cgm_csv(csv_path2)
                cgm_rows += cgm_rows2

            if cgm_rows:
                X, y = [], []
                phases    = list(CYCLE_MULTIPLIERS)
                exercises = list(EXERCISE_MULTIPLIERS)
                stresses  = list(STRESS_MULTIPLIERS)
                for row in cgm_rows:
                    exercise = random.choice(exercises)
                    stress   = random.choice(stresses)
                    phase    = random.choice(phases)
                    carbs    = random.choice([0, 0, 0, 15, 30])
                    iob      = random.uniform(0, 4)
                    features = encode(row['bgl'], row['bgl_trend'],
                                      exercise, stress, phase, carbs, iob)
                    norm_f   = normalise(features)
                    bgl_error  = row['bgl'] - TARGET_BGL
                    cycle_f    = CYCLE_MULTIPLIERS[phase]
                    stress_f   = STRESS_MULTIPLIERS[stress]
                    exercise_f = EXERCISE_MULTIPLIERS[exercise]
                    delivery = (DEFAULT_DELIVERY + BGL_CORRECTION_GAIN * bgl_error
                             + 0.15 * row['bgl_trend']
                             + (cycle_f - 1.0) * 1.5
                             + (stress_f - 1.0) * 0.8
                             - (1.0 - exercise_f) * 1.2
                             + 0.01 * carbs)  # PLACEHOLDER - should be a separate carb bolus, see WARNING above encode()
                    delivery = max(MIN_DELIVERY, min(delivery, MAX_DELIVERY))
                    X.append(norm_f)
                    y.append(normaliseTarget(delivery))
            else:
                X, y = generate_training_data(n_synthetic, seed)
        else:
            X, y = generate_training_data(n_synthetic, seed)

        indices = list(range(len(X)))
        random.shuffle(indices)
        X = [X[i] for i in indices]
        y = [y[i] for i in indices]

        split   = int(len(X) * (1 - test_split))
        X_train, y_train = X[:split], y[:split]
        X_test,  y_test  = X[split:], y[split:]

        print("[TRAIN] {} train / {} test samples".format(len(X_train), len(X_test)))
        print("[TRAIN] Training for {} epochs...".format(epochs))

        self.net.fit(X_train, y_train, epochs=epochs, verbose=True)
        self.trained = True

        r2  = self.net.score_r2(X_test, y_test)
        acc = self.net.accuracy_within(X_test, y_test, tolerance_units=0.5)
        status = "PASS" if acc >= 95 else "FAIL - try more epochs or neurons"
        print("[EVAL]  R2: {:.4f}".format(r2))
        print("[EVAL]  Decision accuracy (+/-0.5 U/h): {:.1f}%  [{}]".format(acc, status))
        return self

    def decide(self, bgl, bgl_trend=0.0, exercise='none', stress='low',
               cycle_phase='follicular', carbs_g=0,
               hours_since_bolus=4.0, cgm_active=True):
        """
        carbs_g: PLACEHOLDER ONLY. Currently folds carb intake into this
        same continuous 5-min correction rather than a proper one-time
        carb bolus - see WARNING above encode(). Leave at 0 until that's
        implemented; a non-zero value will keep inflating every microbolus
        for as long as it's passed, not just around the meal.

        Decision precedence, all of which now ALWAYS return a 'steps' key
        (previously only the AUTO branch computed steps, which meant any
        SAFE-mode result passed to sendMotorCommand() would KeyError or
        require the caller to guess a fallback):

          1. cgm_active is False  -> true sensor dropout -> SAFE (hold SAFE_DELIVERY)
          2. bgl > CGM_VALID_MAX  -> implausible reading, sensor error suspected -> SAFE
          3. bgl <= CGM_VALID_MIN -> genuinely low/at-risk reading -> SUSPEND (0 U/h)
             This is deliberately NOT the same branch as sensor dropout: if the
             CGM is reliably reporting a low value, delivering ANY non-zero
             insulin is the wrong call. Suspend delivery instead.
          4. otherwise            -> AUTO (neural net decides)
        """

        if not cgm_active:
            predicted = SAFE_DELIVERY
            mode      = 'SAFE'
            reason    = 'CGM signal lost - holding safe fallback delivery'

        elif bgl > CGM_VALID_MAX:
            predicted = SAFE_DELIVERY
            mode      = 'SAFE'
            reason    = 'BGL reading implausibly high - sensor error suspected'

        elif bgl <= CGM_VALID_MIN:
            predicted = MIN_DELIVERY
            mode      = 'SUSPEND'
            reason    = 'BGL below safe threshold ({:.1f} < {:.1f}) - insulin suspended'.format(
                            bgl, CGM_VALID_MIN)

        else:
            if not self.trained:
                raise RuntimeError("Call .train() before .decide()")

            features  = encode(bgl, bgl_trend, exercise, stress, cycle_phase,
                               carbs_g, hours_since_bolus)
            norm_f    = normalise(features)
            norm_pred = self.net.predictOne(norm_f)
            predicted = denormaliseTarget(norm_pred)

            bgl_status = ("HIGH"      if bgl > TARGET_BGL + 1.5 else
                          "LOW"       if bgl < TARGET_BGL - 1.0 else
                          "ON TARGET")
            mode   = 'AUTO'
            reason = "BGL {:.1f} ({}), trend {:+.2f}, ex={}, stress={}, phase={}".format(
                        bgl, bgl_status, bgl_trend, exercise, stress, cycle_phase)

        # Hard safety clamp applies regardless of which branch set `predicted` -
        # never trust any single code path (including the network) to self-limit.
        predicted = max(MIN_DELIVERY, min(predicted, MAX_DELIVERY))
        microbolus_units = deliveryToMicrobolus(predicted)
        microbolus_units = roundToIncrement(microbolus_units)  # snap to nearest 0.05U
        steps = microbolusToSteps(microbolus_units)

        return {
            'microbolus_units': round(microbolus_units, 4),  # exact dose (U) delivered THIS cycle, already rounded to the nearest 0.05U
            'delivery_rate':    round(predicted, 3),          # U/h - the hourly-equivalent rate this microbolus was derived from, not itself a dose
            'steps':            steps,                        # motor steps corresponding to microbolus_units
            'mode':             mode,
            'reason':           reason,
        }

    def sendMotorCommand(self, steps):
            if self.serial is None:
                print(f"[SIM] Would send: MOVE {steps}")
                return None

            command = f"MOVE {steps}\n"
            self.serial.write(command.encode())
            response = self.serial.readline().decode().strip()

            if response.startswith("OK"):
                print(f"Confirmed: {response}")
            elif response.startswith("ERR"):
                print(f"Rejected by Arduino: {response}")
            else:
                print(f"Unexpected response: '{response}' - check connection")

            return response

if __name__ == "__main__":
    import sys

    WEIGHTS_FILE = "insalo_weights.json"
    controller = INSALOController()

    if os.path.exists(WEIGHTS_FILE):
        print("[INIT] Loading pre-trained weights from file...")
        controller.load_model(WEIGHTS_FILE)
    else:
        print("[INIT] No saved weights found. Training model now...")
        csv_path = "data/processed/cleaned_medtronic_data.csv"
        controller.train(csv_path=csv_path)
        controller.save_model(WEIGHTS_FILE)

    if len(sys.argv) >= 8:
        bgl = float(sys.argv[1])
        bgl_trend = float(sys.argv[2])
        exercise = sys.argv[3]
        stress = sys.argv[4]
        cycle_phase = sys.argv[5]
        carbs_g = float(sys.argv[6])
        hours_since_bolus = float(sys.argv[7])

        result = controller.decide(
            bgl,
            bgl_trend,
            exercise,
            stress,
            cycle_phase,
            carbs_g,
            hours_since_bolus,
        )

        print("\nDecision Result:")
        print(result)
        controller.sendMotorCommand(result["steps"])

    else:
        print("\nRunning with default test inputs...")
        result = controller.decide(
            bgl=4.0,
            bgl_trend=0.2,
            exercise="moderate",
            stress="low",
            cycle_phase="follicular",
            carbs_g=0,
            hours_since_bolus=1.2,
        )

        print("Decision Result:", result)
        controller.sendMotorCommand(result["steps"])