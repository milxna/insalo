import math
import random
import csv
import os


# =============================================================================
# CONSTANTS
# =============================================================================

TARGET_BGL    = 6.1    # mmol/L 
SAFE_BASAL    = 1.5    # U/h    - AS PER CRITERION 2
DEFAULT_BASAL = 3.0    # U/h    
MAX_BASAL     = 5.0    # U/h    
MIN_BASAL     = 0.0    # U/h    
CGM_VALID_MIN = 4.0    # mmol/L 
CGM_VALID_MAX = 22.0   # mmol/L 


# =============================================================================
# PHYSIOLOGICAL FACTORS
# =============================================================================

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


# =============================================================================
# FEATURE ENCODING
# =============================================================================

def encode(bgl, bgl_trend, exercise, stress, cycle_phase,
           carbs_g=0, hours_since_bolus=4.0):
    
    bgl_error  = bgl - TARGET_BGL                          # how far from target ??? 
    exercise_f = EXERCISE_MULTIPLIERS.get(exercise, 1.0)
    stress_f   = STRESS_MULTIPLIERS.get(stress, 1.0)
    cycle_f    = CYCLE_MULTIPLIERS.get(cycle_phase, 1.0)
    iob        = max(0.0, 1.0 - hours_since_bolus / 4.0)  

    return [bgl, bgl_error, bgl_trend, exercise_f, stress_f,
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


# =============================================================================
# NORMALISATION
# =============================================================================

FEATURE_MINS = [2.0,   -7.0,  -3.0,  0.65,  1.00,  0.90,  0.0,   0.0]
FEATURE_MAXS = [25.0,  12.0,   3.0,  1.00,  1.30,  1.20,  60.0,  1.0]

def normalise(features):
    """Scale each feature to [0, 1] so no single feature dominates training."""
    result = []
    for val, lo, hi in zip(features, FEATURE_MINS, FEATURE_MAXS):
        span = hi - lo
        result.append((val - lo) / span if span > 0 else 0.0)
    return result

def normalise_target(basal):
    """Scale the target (basal rate) to [0, 1] for the output neuron."""
    return (basal - MIN_BASAL) / (MAX_BASAL - MIN_BASAL)

def denormalise_target(norm_basal):
    """Convert the network's [0, 1] output back to a real basal rate."""
    return norm_basal * (MAX_BASAL - MIN_BASAL) + MIN_BASAL


# =============================================================================
# ACTIVATION FUNCTIONS
# 1. ReLU - Rectified Linear Unit (the most common activation in neural networks)
# Basically: "if the signal is positive, pass it through; if negative, block it."
#
# 2. Sigmoid - used for the output layer, squashing the output from [0,1] 
# Formula: sigmoid(x) = 1 / (1 + e^(-x))
# =============================================================================

def relu(x):
    return max(0.0, x)

def relu_derivative(x):
    # Derivative of ReLU — needed for backpropagation. 1 if x>0, else 0.
    return 1.0 if x > 0 else 0.0

def sigmoid(x):
    # Clamp to prevent math overflow for very large/small x
    x = max(-500.0, min(500.0, x))
    return 1.0 / (1.0 + math.exp(-x))

def sigmoid_derivative(sig_x):
    # Derivative of sigmoid given the already-computed sigmoid value.
    return sig_x * (1.0 - sig_x)


# =============================================================================
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
# =============================================================================

class NeuralNetwork:

    def __init__(self, layer_sizes=(N_FEATURES, 16, 8, 1), learning_rate=0.01):
        self.lr = learning_rate
        self.layer_sizes = layer_sizes

        # weights[i] is a 2D list: weights[i][j][k] is the weight from
        # neuron k in layer i to neuron j in layer i+1
        self.weights = []
        self.biases  = []

        for i in range(len(layer_sizes) - 1):
            n_in  = layer_sizes[i]
            n_out = layer_sizes[i + 1]

            # "He initialisation": scale random weights by sqrt(2/n_in)
            # This keeps activations from exploding or vanishing at the start
            scale = math.sqrt(2.0 / n_in)
            layer_w = [[random.gauss(0, scale) for _ in range(n_in)]
                       for _ in range(n_out)]
            layer_b = [0.0] * n_out

            self.weights.append(layer_w)
            self.biases.append(layer_b)

    def forward(self, x):
        activations     = [x]   # layer 0 = the raw input
        pre_activations = []

        for i, (W, b) in enumerate(zip(self.weights, self.biases)):
            prev = activations[-1]
            z = [sum(W[j][k] * prev[k] for k in range(len(prev))) + b[j]
                 for j in range(len(W))]
            pre_activations.append(z)

            # Last layer uses sigmoid; all hidden layers use ReLU
            is_last = (i == len(self.weights) - 1)
            if is_last:
                a = [sigmoid(zi) for zi in z]
            else:
                a = [relu(zi) for zi in z]

            activations.append(a)

        return activations, pre_activations

    def predict_one(self, x):
        """Predict a single normalised basal rate for one input."""
        activations, _ = self.forward(x)
        return activations[-1][0]   # single output neuron

    def predict(self, X):
        """Predict for a list of inputs."""
        return [self.predict_one(x) for x in X]

    # ---- Backpropagation ----------------------------------------------------

    def train_one(self, x, y_true):
        
        activations, pre_activations = self.forward(x)
        prediction = activations[-1][0]

        # output layer error
        output_error = 2.0 * (prediction - y_true)
        sig_deriv    = sigmoid_derivative(prediction)
        deltas = [[output_error * sig_deriv]]   # delta for output layer

        # backpropagate through hidden layers 
        for i in reversed(range(len(self.weights) - 1)):
            W_next   = self.weights[i + 1]   # weights going forward from this layer
            d_next   = deltas[0]             # deltas from the layer ahead
            z_curr   = pre_activations[i]    # pre-activation values at this layer

            # For each neuron in this layer: sum up how much it contributed
            # to all errors in the next layer, then scale by activation derivative
            d_curr = []
            for j in range(len(z_curr)):
                # Sum of (delta_k * weight_k_j) for all neurons k in next layer
                upstream = sum(d_next[k] * W_next[k][j]
                               for k in range(len(d_next)))
                d_curr.append(upstream * relu_derivative(z_curr[j]))

            deltas.insert(0, d_curr)   # prepend so deltas[0] = first hidden layer

        # ---- Step 3: update weights and biases with gradient descent ---------
        for i in range(len(self.weights)):
            prev_activations = activations[i]
            for j in range(len(self.weights[i])):
                for k in range(len(self.weights[i][j])):
                    # Weight update: w -= learning_rate * gradient
                    self.weights[i][j][k] -= self.lr * deltas[i][j] * prev_activations[k]
                self.biases[i][j] -= self.lr * deltas[i][j]

        # Return loss for monitoring
        return (prediction - y_true) ** 2

    # ---- Training Loop -------------------------------------------------------

    def fit(self, X, y, epochs=200, verbose=True):
        """
        Train the network over multiple passes through the data.
        Each full pass through all training examples is called an "epoch."
        """
        n = len(X)
        for epoch in range(epochs):
            # Shuffle data each epoch so the network doesn't memorise order
            indices = list(range(n))
            random.shuffle(indices)

            total_loss = 0.0
            for i in indices:
                loss = self.train_one(X[i], y[i])
                total_loss += loss

            avg_loss = total_loss / n

            # Print progress every 50 epochs
            if verbose and (epoch + 1) % 50 == 0:
                print("  Epoch {:>3}/{}  |  avg loss: {:.6f}".format(
                    epoch + 1, epochs, avg_loss))

        return self

    # ---- Evaluation ----------------------------------------------------------

    def score_r2(self, X, y):
        """R-squared: 1.0 = perfect predictions, 0.0 = no better than the mean."""
        preds  = self.predict(X)
        mean_y = sum(y) / len(y)
        ss_res = sum((p - t) ** 2 for p, t in zip(preds, y))
        ss_tot = sum((t - mean_y) ** 2 for t in y)
        return 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    def accuracy_within(self, X, y_norm, tolerance_units=0.5):
        """
        Evaluation Criterion 3: % of predictions within +/-tolerance U/h.
        Converts back from normalised [0,1] to real U/h for comparison.
        """
        preds   = self.predict(X)
        correct = 0
        for p, t in zip(preds, y_norm):
            p_real = denormalise_target(p)
            t_real = denormalise_target(t)
            if abs(p_real - t_real) <= tolerance_units:
                correct += 1
        return 100.0 * correct / len(y_norm)


# =============================================================================
# SECTION 7 - SYNTHETIC TRAINING DATA
#
# The ground-truth formula encodes the clinical relationships from Criterion 1.
# The network learns to approximate this formula purely from examples —
# it never sees the formula itself. That's the core of machine learning.
# =============================================================================

def generate_training_data(n=2000, seed=42):
    """Generate n synthetic (features, basal_rate) pairs for training."""
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

        # Ground-truth clinical formula (what we're teaching the network)
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
        y.append(normalise_target(basal))

    return X, y


# =============================================================================
# SECTION 8 - CSV LOADER (for your real Medtronic data)
# =============================================================================

def load_cgm_csv(filepath):
    """Load CGM readings from your Medtronic CSV export."""
    if not os.path.exists(filepath):
        print("[WARN] CSV not found at: {}".format(filepath))
        return []

    rows = []
    with open(filepath, newline='', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for row in reader:
            bgl_raw = (row.get('SGValue') or row.get('SG Value') or
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


# =============================================================================
# SECTION 9 - MAIN CONTROLLER
# =============================================================================

class INSALOController:
    """
    Top-level controller. Instantiate on the Raspberry Pi Zero W.

    Usage:
        ctrl = INSALOController()
        ctrl.train()
        result = ctrl.decide(bgl=9.5, bgl_trend=1.2, exercise='moderate',
                             stress='high', cycle_phase='luteal')
    """

    def __init__(self):
        self.net     = NeuralNetwork(layer_sizes=(N_FEATURES, 16, 8, 1),
                                     learning_rate=0.01)
        self.trained = False

    def train(self, csv_path=None, n_synthetic=2000, epochs=300,
              test_split=0.2, seed=42):
        """
        Train the neural network.
        Uses real Medtronic BGL data if CSV is provided; otherwise synthetic.
        """
        random.seed(seed)
        print("\n[TRAIN] Preparing data...")

        if csv_path and os.path.exists(csv_path):
            cgm_rows = load_cgm_csv(csv_path)
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
                    y.append(normalise_target(basal))
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
        """
        Make one insulin delivery decision. Call every 5 minutes.
        Returns: { 'basal_rate': float, 'mode': str, 'reason': str }
        """
        # Evaluation Criterion 2: Safe Mode
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
        norm_pred = self.net.predict_one(norm_f)
        predicted = denormalise_target(norm_pred)
        predicted = max(MIN_BASAL, min(predicted, MAX_BASAL))  # safety clamp

        bgl_status = ("HIGH"      if bgl > TARGET_BGL + 1.5 else
                      "LOW"       if bgl < TARGET_BGL - 1.0 else
                      "ON TARGET")

        return {
            'basal_rate': round(predicted, 3),
            'mode':       'AUTO',
            'reason':     "BGL {:.1f} ({}), trend {:+.2f}, ex={}, stress={}, phase={}".format(
                              bgl, bgl_status, bgl_trend, exercise, stress, cycle_phase),
        }


# =============================================================================
#  DEMO / TEST SCRIPT
# =============================================================================

# def run_demo():
#     print("=" * 65)
#     print("  INSALO - Machine Learning Base | Neural Network (MLP)")
#     print("  VCE Systems Engineering 3&4 2026 | Milana Kumykova")
#     print("=" * 65)

#     ctrl = INSALOController()

#     csv_path = os.path.join("data", "processed", "cleaned_medtronic_data.csv")
#     ctrl.train(csv_path=csv_path if os.path.exists(csv_path) else None)

#     # ---- Evaluation Criterion 3 - Decision Accuracy -------------------------
#     print("\n---- Evaluation Criterion 3 - Scenarios ----")
#     scenarios = [
#         ("High exercise",            "high",     "low",    "follicular", 0,  4, 7.0),
#         ("Low exercise",             "light",    "low",    "follicular", 0,  4, 7.0),
#         ("High stress",              "none",     "high",   "follicular", 0,  4, 7.0),
#         ("Luteal phase",             "none",     "low",    "luteal",     0,  4, 7.0),
#         ("Ovulation phase",          "none",     "low",    "ovulation",  0,  4, 7.0),
#         ("High BGL + stress + meal", "none",     "high",   "luteal",     30, 2, 12.0),
#         ("Low BGL + high exercise",  "high",     "low",    "menstrual",  15, 1, 4.5),
#     ]
#     for label, ex, st, ph, carbs, iob, bgl in scenarios:
#         r = ctrl.decide(bgl=bgl, bgl_trend=0.0, exercise=ex, stress=st,
#                         cycle_phase=ph, carbs_g=carbs, hours_since_bolus=iob)
#         print("  {:<35} -> {:.3f} U/h  [{}]".format(label, r['basal_rate'], r['mode']))

#     # ---- Evaluation Criterion 2 - Safe Mode ---------------------------------
#     print("\n---- Evaluation Criterion 2 - Safe Mode ----")
#     r1 = ctrl.decide(bgl=7.0, cgm_active=False)
#     r2 = ctrl.decide(bgl=1.2)
#     print("  CGM offline              -> {} U/h  [{}]".format(r1['basal_rate'], r1['mode']))
#     print("  BGL 1.2 (out of range)   -> {} U/h  [{}]".format(r2['basal_rate'], r2['mode']))

#     # ---- Evaluation Criterion 4 - Adaptive Rising BGL -----------------------
#     print("\n---- Evaluation Criterion 4 - Rising BGL Sequence ----")
#     for bgl_val, trend in [(7.0, 0.0), (9.5, 1.2), (12.0, 1.5), (16.5, 2.0)]:
#         r = ctrl.decide(bgl=bgl_val, bgl_trend=trend)
#         print("  BGL {:.1f} mmol/L, trend {:+.1f} -> {:.3f} U/h".format(
#             bgl_val, trend, r['basal_rate']))

#     print("\nDone.")


# if __name__ == "__main__":
#     run_demo()