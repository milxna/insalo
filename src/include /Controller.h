#pragma once

#include "constants/Units.h"

class PumpController {
public:
    PumpController(double syringeArea,
                   double leadScrewPitch,
                   int stepsPerRevolution,
                   int microstepping);

    void deliver(milliliter_t volume, second_t time);

private:
    double volumeToSteps(milliliter_t volume) const;

    double syringeArea_;
    double leadScrewPitch_;
    int stepsPerRevolution_;
    int microstepping_;
};
