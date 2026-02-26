#pragma once

#include "constants/Constants.h"

struct StepperConfig {
    int stepsPerRevolution = StepperConstants::stepsPerRevolution;
    int microstepping = StepperConstants::microstepping;
    double leadScrewPitchMm = StepperConstants::leadScrewPitchMm;
    double minStepsPerSecond = StepperConstants::minStepsPerSecond;
    double maxStepsPerSecond = StepperConstants::maxStepsPerSecond;
};

class Stepper {
public:
    Stepper();
    explicit Stepper(const StepperConfig& config);

    void configure(const StepperConfig& config);

    void moveSteps(int steps, double steps_per_second);
    double stepsPerMillimeter() const;

    int lastCommandedSteps() const;
    double lastCommandedSpeed() const;

private:
    StepperConfig config_{};
    int lastCommandedSteps_ = 0;
    double lastCommandedSpeed_ = 0.0;
};
