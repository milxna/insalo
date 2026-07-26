import math
import random
import csv
import os
import serial
import json

# CONSTANTS
TARGET_BGL    = 6.1    # mmol/L 
SAFE_BASAL    = 1.5    # U/h    - AS PER CRITERION 2
DEFAULT_BASAL = 3.0    # U/h    
MAX_BASAL     = 5.0    # U/h    
MIN_BASAL     = 0.0    # U/h    
CGM_VALID_MIN = 4.0    # mmol/L 
CGM_VALID_MAX = 22.0   # mmol/L 

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
    "light":    0.9,   # walking, yoga (10% decrease in insulin)
    "moderate": 0.8,   # swimming, cycling (20% decrease in insulin) 
    "high":     0.65,   # AFL, HIIT, contact sport (35% decrease in insulin)
}

STRESS_MULTIPLIERS = {
    "low":    1.00, #standard insulin delivery
    "medium": 1.05, #5% increase in insulin
    "high":   1.1, #10% increase in insulin need 
}


# FEATURE ENCODING

def encode(bgl, bglTrend, exercise, stress, cyclePhase,
           carbs_g=0, hoursSinceBolus=4.0):
    
    bgl_error  = bgl - TARGET_BGL                          # how far from target ??? 
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
    "Carbs (g)",
    "Insulin on board",
]

N_FEATURES = len(FEATURE_NAMES)


# NORMALISATION - make everything scaled to the same value for what it is. (see folio for explanation)

featureMins = [2.0,   -7.0,  -3.0,  0.65,  1.00,  0.90,  0.0,   0.0]
featureMaxs = [25.0,  12.0,   3.0,  1.00,  1.30,  1.20,  60.0,  1.0]

def normalise(features):
    result = []
    for val, lo, hi in zip(features, featureMins, featureMaxs):
        span = hi - lo
        result.append((val - lo) / span if span > 0 else 0.0)
    return result

def normaliseTarget(basal):
    return (basal - MIN_BASAL) / (MAX_BASAL - MIN_BASAL)

def denormaliseTarget(norm_basal):
    return norm_basal * (MAX_BASAL - MIN_BASAL) + MIN_BASAL

def basalToSteps(basalRate):
    units = basalRate * CONTROL_PERIOD_MIN
    return round(units * STEPS_PER_UNIT)

# DEFINE ACTIVATION FUNCTIONS
# 1. ReLU - Rectified Linear Unit (the most common activation in neural networks)
# Basically: "if the signal is positive, pass it through; if negative, block it."
# 2. Sigmoid - used for the output layer, squashing the output from [0,1] 
# Formula: sigmoid(x) = 1 / (1 + e^(-x))

def relu(x):
    return max(0.0, x)

def reluDerivative(x):
    # Derivative of ReLU — needed for backpropagation. 1 if x>0, else 0.
    return 1.0 if x > 0 else 0.0

def sigmoid(x):
    # Clamp to prevent math overflow for very large/small x
    x = max(-500.0, min(500.0, x))
    return 1.0 / (1.0 + math.exp(-x))

def sigmoidDerivative(sig_x):
    # Derivative of sigmoid given the already-computed sigmoid value.
    return sig_x * (1.0 - sig_x)


# THE NEURAL NETWORK
# Architecture: 8 inputs -> 16 neurons -> 8 neurons -> 1 output
#
# Layer 1 (hidden): 16 neurons with ReLU activation : learns low level patterns
#
# Layer 2 (hidden): 8 neurons with ReLU activation : combines patterns
#
# Layer 3 (output): 1 neuron with Sigmoid activation : outputs a value of [0,1]
#
# Each neuron has a weight and bias, which are adjusted during training to minimise the error between
# the predicted basal rate and the true basal rate (from our synthetic data).

class NeuralNetwork:

    def __init__(self, layerSizes=(N_FEATURES, 16, 8, 1), learningRate=0.01):
        self.lr = learningRate
        self.layer_sizes = layerSizes

        # weights[i] is a 2D list: weights[i][j][k] is the weight from
        # neuron k in layer i to neuron j in layer i+1
        self.weights = []
        self.biases  = []

        for i in range(len(layerSizes) - 1):
            n_in  = layerSizes[i]
            n_out = layerSizes[i + 1]

            # "He initialisation": scale random weights by sqrt(2/n_in)
            scale = math.sqrt(2.0 / n_in)
            layer_w = [[random.gauss(0, scale) for _ in range(n_in)]
                       for _ in range(n_out)]
            layer_b = [0.0] * n_out

            self.weights.append(layer_w)
            self.biases.append(layer_b)

# Forward Pass
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

# Backpropagation

    def trainOne(self, x, y_true):
        
        activations, pre_activations = self.forward(x)
        prediction = activations[-1][0]

        # output layer error
        output_error = 2.0 * (prediction - y_true)
        sig_deriv    = sigmoidDerivative(prediction)
        deltas = [[output_error * sig_deriv]]   

        # backpropagate through hidden layers 
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

        # update weights and biases
        for i in range(len(self.weights)):
            prev_activations = activations[i]
            for j in range(len(self.weights[i])):
                for k in range(len(self.weights[i][j])):
                    # Weight update: w -= learning_rate * gradient
                    self.weights[i][j][k] -= self.lr * deltas[i][j] * prev_activations[k]
                self.biases[i][j] -= self.lr * deltas[i][j]

        # Return loss for monitoring
        return (prediction - y_true) ** 2

# training!

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

            # print progress every 50 epochs
            if verbose and (epoch + 1) % 50 == 0:
                print("  Epoch {:>3}/{}  |  avg loss: {:.6f}".format(
                    epoch + 1, epochs, avg_loss))

        return self

#self evaluation 

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
        """Saves weights, biases, and layer configuration to a JSON file."""
        data = {
            "layer_sizes": self.layer_sizes,
            "weights": self.weights,
            "biases": self.biases,
        }
        with open(filepath, "w") as f:
            json.dump(data, f)
        print(f"[INFO] Weights successfully saved to {filepath}")

    def load_weights(self, filepath="model_weights.json"):
        """Loads weights, biases, and layer configuration from a JSON file."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Weight file not found: {filepath}")

        with open(filepath, "r") as f:
            data = json.load(f)

        self.layer_sizes = data["layer_sizes"]
        self.weights = data["weights"]
        self.biases = data["biases"]
        print(f"[INFO] Weights successfully loaded from {filepath}")


# SYNTHETISE DATA 
# because I don't have unlimited real data I need something for it to train on, so this basically generates that

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

        basal = (DEFAULT_BASAL
                 + 0.40 * bgl_error
                 + 0.15 * trend
                 + (cycle_f  - 1.0) * 1.5
                 + (stress_f - 1.0) * 0.8
                 - (1.0 - exercise_f) * 1.2
                 + 0.01 * carbs)
        basal = max(MIN_BASAL, min(basal, MAX_BASAL))

        X.append(norm_feats)
        y.append(normaliseTarget(basal))

    return X, y


# LOAD CSV 
# the real data I've fed it

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
    #built to run on Raspberry Pi Zero W 

    #fixme
    # def __init__(self):
    #     # self.serial = serial.Serial(
    #     #                 "/dev/ttyACM0", also this depends ont he port of the arduino so this is subj to change
    #     #                 115200,
    #     #                 timeout=1)
    #     self.serial = None #placeholder
    #     self.net     = NeuralNetwork(layerSizes=(N_FEATURES, 16, 8, 1),
    #                                  learningRate=0.01)
    #     self.trained = False

    def __init__(self):
        self.serial = None
        self.net = NeuralNetwork(
            layerSizes=(N_FEATURES, 16, 8, 1), learningRate=0.01
        )
        self.trained = False

    def save_model(self, filepath="insalo_weights.json"):
        if not self.trained:
            print("[WARN] Attempting to save weights, but model is not trained yet.")
        self.net.save_weights(filepath)

    def load_model(self, filepath="insalo_weights.json"):
        self.net.load_weights(filepath)
        self.trained = True  # Mark as trained so decide() works immediately

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
                    basal = (DEFAULT_BASAL + 0.40 * bgl_error
                             + 0.15 * row['bgl_trend']
                             + (cycle_f - 1.0) * 1.5
                             + (stress_f - 1.0) * 0.8
                             - (1.0 - exercise_f) * 1.2
                             + 0.01 * carbs)
                    basal = max(MIN_BASAL, min(basal, MAX_BASAL))
                    X.append(norm_f)
                    y.append(normaliseTarget(basal))
            else:
                X, y = generate_training_data(n_synthetic, seed)
        else:
            X, y = generate_training_data(n_synthetic, seed)

        # Shuffle and split
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

        # safe mode !!!!!
        if not cgm_active or not (CGM_VALID_MIN <= bgl <= CGM_VALID_MAX):
            return {
                'basal_rate': SAFE_BASAL,
                'mode':       'SAFE',
                'reason':     'CGM dropout or invalid BGL - safe mode active',
            }

        if not self.trained:
            raise RuntimeError("Call .train() before .decide()")

        features  = encode(bgl, bgl_trend, exercise, stress, cycle_phase,
                           carbs_g, hours_since_bolus)
        norm_f    = normalise(features)
        norm_pred = self.net.predictOne(norm_f)
        predicted = denormaliseTarget(norm_pred)
        predicted = max(MIN_BASAL, min(predicted, MAX_BASAL))  # safety clamp

        bgl_status = ("HIGH"      if bgl > TARGET_BGL + 1.5 else
                      "LOW"       if bgl < TARGET_BGL - 1.0 else
                      "ON TARGET")

        steps = basalToSteps(predicted)
        return {
            'basal_rate': round(predicted, 3),
            'steps':      steps,
            'mode':       'AUTO',
            'reason':     "BGL {:.1f} ({}), trend {:+.2f}, ex={}, stress={}, phase={}".format(
                            bgl, bgl_status, bgl_trend, exercise, stress, cycle_phase),
        }

#test code placeholder
def sendMotorCommand(steps):
    print(f"Sending to Arduino: MOVE {steps}")


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
        sendMotorCommand(result["steps"])

    else:
        print("\n[TEST] Running with default test inputs...")
        result = controller.decide(
            bgl=10.5,
            bgl_trend=0.2,
            exercise="moderate",
            stress="low",
            cycle_phase="follicular",
            carbs_g=40,
            hours_since_bolus=1.2,
        )

        print("Decision Result:", result)
        sendMotorCommand(result["steps"])

    # #printing basal rate for c++
    # print(result['basal_rate']) 
    # deprecated as no longer using c++ for the purposes of this project

# def sendMotorCommand(steps):
#     arduino = serial.Serial(
#         "/dev/cu.usbmodemXXXX",
#         115200,
#         timeout=1
#     )
#     command = f"MOVE {steps}\n"
#     arduino.write(command.encode())
#     response = arduino.readline().decode().strip()
#     print(response)
#     arduino.close()



