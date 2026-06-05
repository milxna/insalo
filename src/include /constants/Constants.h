#pragma once

#include "Units.h"

namespace PumpConstants {
using namespace units::literal;
inline constexpr milliliters_per_second_t insulinFlowRatePerSecond = 0.02_mL / 1_s;
}

namespace StepperConstants {
inline constexpr int    stepsPerRevolution  = 200;
inline constexpr int    microstepping       = 16;
inline constexpr double leadScrewPitchMm    = 2.0;
inline constexpr double minStepsPerSecond   = 1.0;
inline constexpr double maxStepsPerSecond   = 5000.0;

constexpr int PIN_STEP = 17;
constexpr int PIN_DIR  = 27;
constexpr int PIN_EN   = 22;
constexpr int PIN_MS1  = 10;
constexpr int PIN_MS2  = 9;
constexpr int PIN_MS3  = 11;
}