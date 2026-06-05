#pragma once

#include "constants/Constants.h"

#ifdef __linux__
#include <pigpio.h>
#endif

class Stepper {
public:
    Stepper();
    explicit Stepper(const StepperConfig& config);

    void configure(const StepperConfig& config);
    void init();
    void shutdown();

    void moveSteps(int steps, double stepsPerSecond);
    double stepsPerMillimeter() const;

    int lastCommandedSteps() const;
    double lastCommandedSpeed() const;

private:
    StepperConfig config_{};
    int lastCommandedSteps_ = 0;
    double lastCommandedSpeed_ = 0.0;
};


struct StepperConfig {
    int stepsPerRevolution  = StepperConstants::stepsPerRevolution;
    int microstepping       = StepperConstants::microstepping;
    double leadScrewPitchMm = StepperConstants::leadScrewPitchMm;
    double minStepsPerSecond = StepperConstants::minStepsPerSecond;
    double maxStepsPerSecond = StepperConstants::maxStepsPerSecond;

    int pinStep = StepperConstants::PIN_STEP;
    int pinDir  = StepperConstants::PIN_DIR;
    int pinEn   = StepperConstants::PIN_EN;
    int pinMs1  = StepperConstants::PIN_MS1;
    int pinMs2  = StepperConstants::PIN_MS2;
    int pinMs3  = StepperConstants::PIN_MS3;
};
