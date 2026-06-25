#include "Controller.h"

PumpController::PumpController(double syringeArea,
                               double leadScrewPitch,
                               int stepsPerRevolution,
                               int microstepping)
    : syringeArea_(syringeArea),
      leadScrewPitch_(leadScrewPitch),
      stepsPerRevolution_(stepsPerRevolution),
      microstepping_(microstepping) {}

void PumpController::deliver(milliliter_t volume, second_t time) {
    double steps = volumeToSteps(volume);
    double stepsPerSecond = steps / time.value;
    // TODO: call stepper.moveSteps(steps, stepsPerSecond)
}

double PumpController::volumeToSteps(milliliter_t volume) const {
    double distanceMm = volume.value / syringeArea_;
    double revolutions = distanceMm / leadScrewPitch_;
    return revolutions * stepsPerRevolution_ * microstepping_;
}