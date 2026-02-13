#include "Units.h"
#include "Controller.h"

double PumpController::volumeToSteps(milliliter_t volume);
{
    double volume = volume.value * 1000.0;
    double travel = volume / syringeArea;
    double revolution = travel / leadScrewPitch;
    int totalSteps = revolutions * stepsPerRevolution * microstepping;

    return totalSteps;
}
