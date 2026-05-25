#pragma once
#include <constants/Constants.h>
#include <constants/Units.h>

class StepperMotor {
public:
    StepperMotor();
    ~StepperMotor();

    void init();
    void enable();
    void disable();
    void driveFromNN(double nnOutput, int stepsToTake = -1);
    void driveToAngle(double targetDegrees, double speedFraction = 0.5);
    void shutdown();

    double currentAngle() const;
    double currentSpeed() const;

private:
    bool running;
    double speed;
    double currentAngleDeg;
};

double runNeuralNetwork(const double* inputs, int n_inputs);