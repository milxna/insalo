"""can be used to emu"""


import random
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

TARGET_GLUCOSE_LOW = 5.5       # lower bound of target range
TARGET_GLUCOSE_HIGH = 6.7      # upper bound of target range
BASAL_RATE = 1.2                # U/hr
BOLUS_THRESHOLD = 8.9           # U
LOW_THRESHOLD = 4.1             # mmol/L
PREDICTIVE_HYPO_TRIGGER = 3.7   #mmol/L
BASE_INSULIN_SENSITIVITY = 2.3  
INSULIN_ACTION_DURATION = 30
CARB_RATIO = 7

# Controller gains
Kp = 0.15
Ki = 0.002
MAX_CORRECTION_PER_DOSE = 1.3
STEP_MINUTES = 5

time_step = 0

MEAL_STEPS = [12, 36]
MEAL_CARBS = [60, 45]

glucose = 3.2
insulin_on_board = 0.0
insulin_history = []
data_log = []

last_dose_time = None
last_dose_amount = 0.0

stress_level = 1     # constants, but i can change 
activity_level = 4   

integral_error = 0.0


def compute_correction_dose(current_glucose):
    global integral_error

    # stops from correcting if im in or below range 
    if current_glucose <= TARGET_GLUCOSE_HIGH:
        return 0.0

    target_mid = (TARGET_GLUCOSE_LOW + TARGET_GLUCOSE_HIGH) / 2
    error = current_glucose - target_mid
    integral_error += error * (STEP_MINUTES / 60.0)
    control_signal = Kp * error + Ki * integral_error

    sensitivity = BASE_INSULIN_SENSITIVITY
    desired_units = control_signal / sensitivity
    desired_units = max(0.0, min(desired_units, MAX_CORRECTION_PER_DOSE))
    return desired_units


def simulate_glucose_change(g, step_index, carbs):
    eff_ins = sum(
        dose["amount"] * max(0, 1 - ((step_index - dose["time"]) * STEP_MINUTES) / INSULIN_ACTION_DURATION)
        for dose in insulin_history
        if (step_index - dose["time"]) * STEP_MINUTES < INSULIN_ACTION_DURATION
    )
    insulin_effect_rate = 1.0 / (INSULIN_ACTION_DURATION / STEP_MINUTES)
    glucose_change_from_insulin = - eff_ins * BASE_INSULIN_SENSITIVITY * insulin_effect_rate if eff_ins > 0 else 0
    glucose_change_from_carbs = carbs * 0.05
    random_fluctuation = random.uniform(-0.03, 0.03)
    baseline_drift = (TARGET_GLUCOSE_HIGH - g) * 0.01
    g += glucose_change_from_insulin + glucose_change_from_carbs + random_fluctuation + baseline_drift
    return max(2.8, min(g, 15.0))

def loop_step():
    global glucose, insulin_on_board, time_step, carb_queue

    # pre-bolus (recommended endocrinologist advice to lower bgls)
    if (time_step + 1) in MEAL_STEPS:
        idx = MEAL_STEPS.index(time_step + 1)
        meal_carbs = MEAL_CARBS[idx]
        premeal_dose = min(meal_carbs / CARB_RATIO, MAX_CORRECTION_PER_DOSE)
        insulin_history.append({"time": time_step, "amount": premeal_dose, "type": "correction"})
        print(f"💉 Pre-meal bolus={premeal_dose:.2f}U for upcoming meal {meal_carbs}g carbs")

    # scheduled meal 
    meal_flag = False
    if time_step in MEAL_STEPS:
        idx = MEAL_STEPS.index(time_step)
        meal_carbs = MEAL_CARBS[idx]
        meal_flag = True
        carb_queue.append({"amount": meal_carbs, "steps_remaining": 16})
        print(f"🍽 Meal at step {time_step}, carbs={meal_carbs}g (absorbing over 80 min)")

    # hypo predicted (suspend before low)
    pred_glucose = glucose  # simplified deterministic prediction
    suspend_basal = False
    if pred_glucose < PREDICTIVE_HYPO_TRIGGER or glucose < LOW_THRESHOLD:
        hypo_carbs = random.choice([30, 35, 40])
        carb_queue.append({"amount": hypo_carbs, "steps_remaining": 6})
        suspend_basal = True
        print(f"⚠️ Predicted low {pred_glucose:.1f} or current low {glucose:.1f}, fast carbs added, basal suspended")

    # basal 
    basal_dose = 0.0
    if not suspend_basal:
        basal_dose = BASAL_RATE / (60 / STEP_MINUTES)
        insulin_history.append({"time": time_step, "amount": basal_dose, "type": "basal"})

    # correction boluses
    correction_dose = compute_correction_dose(glucose)
    if correction_dose > 0:
        insulin_history.append({"time": time_step, "amount": correction_dose, "type": "correction"})
        print(f"💉 Correction dose={correction_dose:.2f}U (glucose={glucose:.2f})")

    # carb absorption/digestion
    carbs_this_step = 0.0
    for c in carb_queue:
        absorbed = c["amount"] / c["steps_remaining"] * 1.1
        carbs_this_step += absorbed
        c["amount"] -= absorbed
        c["steps_remaining"] -= 1
    carb_queue[:] = [c for c in carb_queue if c["steps_remaining"] > 0]

    # glucose change
    glucose = simulate_glucose_change(glucose, time_step, carbs_this_step)
    insulin_on_board = sum(
        dose["amount"] * max(0, 1 - ((time_step - dose["time"]) * STEP_MINUTES) / INSULIN_ACTION_DURATION)
        for dose in insulin_history
        if (time_step - dose["time"]) * STEP_MINUTES < INSULIN_ACTION_DURATION
    )

    # change over time
    data_log.append({
        "time": time_step,
        "glucose": glucose,
        "insulin_on_board": insulin_on_board,
        "stress": stress_level,
        "activity": activity_level,
        "carbs": carbs_this_step,
        "basal": basal_dose,
        "correction": correction_dose,
        "meal_flag": meal_flag
    })

    print(f"Time {time_step:02d} | Glucose {glucose:.1f} | Carbs {carbs_this_step:.1f} | Correction {correction_dose:.2f} | Basal {basal_dose:.2f}")

    time_step += 1

# running simulation
if __name__ == "__main__":
    SIM_STEPS = 72
    carb_queue = []
    print("Starting simulation...\n")
    for _ in range(SIM_STEPS):
        loop_step()
        time.sleep(0.05)

    # plotting results (bgls)
df = pd.DataFrame(data_log)
plt.figure(figsize=(12,6))
plt.plot(df["time"], df["glucose"], label="Glucose (mmol/L)", color="blue")

# plotting meals
meal_times = df[df["meal_flag"]==True]["time"]
for t in meal_times:
    plt.axvline(t, color="orange", linestyle="--", label="Meal" if t==meal_times.iloc[0] else "")

# plot correction insulin events
correction_times = df[df["correction"]>0]["time"]
for t in correction_times:
    plt.axvline(t, color="red", linestyle=":", label="Correction" if t==correction_times.iloc[0] else "")

# target glucose lines
plt.axhline(TARGET_GLUCOSE_LOW, color="green", linestyle="--", label=f"Target Low {TARGET_GLUCOSE_LOW}")
plt.axhline(TARGET_GLUCOSE_HIGH, color="green", linestyle="--", label=f"Target High {TARGET_GLUCOSE_HIGH}")

# fixed y scale
plt.ylim(2.8, 22)

plt.xlabel("Time (mins)")
plt.ylabel("Glucose (mmol/L)")
plt.title("INSALO Simulation")
plt.legend()
plt.show()
