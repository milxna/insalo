#include "Controller.h"

PumpController::PumpController(double syringeArea,
                               double leadScrewPitch,
                               int stepsPerRevolution,
                               int microstepping)
    : syringeArea_(syringeArea),
      leadScrewPitch_(leadScrewPitch),
      stepsPerRevolution_(stepsPerRevolution),
      microstepping_(microstepping) {}

double PumpController::volumeToSteps(milliliter_t volume) const {
    const double volumeMm3 = volume.value * 1000.0;
    const double travelMm = volumeMm3 / syringeArea_;
    const double revolutions = travelMm / leadScrewPitch_;
    const double totalSteps = revolutions * stepsPerRevolution_ * microstepping_;

    return totalSteps;
}

void PumpController::deliver(milliliter_t volume, second_t time) {
    (void)time;
    (void)volumeToSteps(volume);
}
