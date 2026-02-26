#include "Stepper.h"

#include <cmath>
#include <stdexcept>

Stepper::Stepper() = default;

Stepper::Stepper(const StepperConfig& config) {
    configure(config);
}

void Stepper::configure(const StepperConfig& config) {
    if (config.stepsPerRevolution <= 0) {
        throw std::invalid_argument("stepsPerRevolution must be positive");
    }
    if (config.microstepping <= 0) {
        throw std::invalid_argument("microstepping must be positive");
    }
    if (config.leadScrewPitchMm <= 0.0) {
        throw std::invalid_argument("leadScrewPitchMm must be positive");
    }
    if (config.minStepsPerSecond <= 0.0) {
        throw std::invalid_argument("minStepsPerSecond must be positive");
    }
    if (config.maxStepsPerSecond < config.minStepsPerSecond) {
        throw std::invalid_argument("maxStepsPerSecond must be >= minStepsPerSecond");
    }

    config_ = config;
}

void Stepper::moveSteps(int steps, double steps_per_second) {
    if (steps == 0) {
        lastCommandedSteps_ = 0;
        lastCommandedSpeed_ = 0.0;
        return;
    }

    const double speedMagnitude = std::abs(steps_per_second);
    if (speedMagnitude < config_.minStepsPerSecond || speedMagnitude > config_.maxStepsPerSecond) {
        throw std::out_of_range("steps_per_second outside configured range");
    }

    lastCommandedSteps_ = steps;
    lastCommandedSpeed_ = steps_per_second;
}

double Stepper::stepsPerMillimeter() const {
    return static_cast<double>(config_.stepsPerRevolution * config_.microstepping) /
           config_.leadScrewPitchMm;
}

int Stepper::lastCommandedSteps() const {
    return lastCommandedSteps_;
}

double Stepper::lastCommandedSpeed() const {
    return lastCommandedSpeed_;
}
