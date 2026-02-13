#pragma once

#include "Units.h"

using namespace units::literal;

class PumpController {
    public:
    PumpController(double syringeArea,
                   double leadScrewPitch, //pitch is the converter between rotational and linear movement 
                   int stepsPerRevolution,
                   int microstepping);

    void deliver(milliliter_t volume,
                second_t time);

    private:
    double volumeToSteps(milliliter_t volume);
};
