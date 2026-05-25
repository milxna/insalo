#pragma once

#include "Units.h"

namespace PumpConstants {
using namespace units::literal;
inline constexpr milliliters_per_second_t insulinFlowRatePerSecond = 0.02_mL / 1_s;
}

namespace StepperConstants {
inline constexpr int stepsPerRevolution(200); // 1.8° per step as per NEMA 17
inline constexpr int microstepping(16);               
inline constexpr int microstepsPerRevolution(stepsPerRevolution * microstepping);
inline constexpr double maxSpeedMicroSteps(3200.0);
inline constexpr unsigned int stepPulseMicroseconds(5);

//pin ids
constexpr int PIN_STEP = 17;
constexpr int PIN_DIR  = 27;
constexpr int PIN_EN   = 22;
constexpr int PIN_MS1  = 10;
constexpr int PIN_MS2  = 9;
constexpr int PIN_MS3  = 11;

}
