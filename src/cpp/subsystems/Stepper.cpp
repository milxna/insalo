#include <subsystems/Stepper.h>

#include <.vscode/pigpio.h>
#include <iostream>
#include <cmath>
#include <algorithm>
#include <stdexcept>
#include <chrono>
#include <thread>
 
using namespace StepperConstants;
  
StepperMotor::StepperMotor()
    : running(false), speed(0.0), currentAngleDeg(0.0) {}
 
StepperMotor::~StepperMotor() {
    shutdown();
}
 
 
void StepperMotor::init() {
    if (gpioInitialise() < 0)
        throw std::runtime_error("init failed, run with sudo");
 
    for (int pin : {PIN_STEP, PIN_DIR, PIN_EN, PIN_MS1, PIN_MS2, PIN_MS3})
        gpioSetMode(pin, PI_OUTPUT);
 
    gpioWrite(PIN_MS1, 1);
    gpioWrite(PIN_MS2, 1);
    gpioWrite(PIN_MS3, 1);
 
    disable();
    running = true;
    std::cout << "[Stepper] Ready. " << microstepsPerRevolution << " µsteps/rev\n";
}
 
void StepperMotor::enable() {
    gpioWrite(PIN_EN, 0);   
}
 
void StepperMotor::disable() {
    gpioWrite(PIN_EN, 1);   
}
 
void StepperMotor::driveFromNN(double nnOutput, int stepsToTake) {
    nnOutput = std::clamp(nnOutput, -1.0, 1.0);
 
    int direction = (nnOutput >= 0.0) ? 1 : 0;
    gpioWrite(PIN_DIR, direction);
    gpioDelayMicroseconds(2);   
 
    double calculatedSpeed = std::abs(nnOutput) * maxSpeedMicroSteps;
    speed = calculatedSpeed;
 
    if (speed < 1.0) {
        std::cout << "[Stepper] NN ≈ 0 → idle\n";
        return;
    }
 
    unsigned int periodUs = static_cast<unsigned int>(1'000'000.0 / speed);
    unsigned int delayUs  = periodUs > stepPulseMicroseconds ? periodUs - stepPulseMicroseconds : 1;
 
    if (stepsToTake < 0) stepsToTake = microstepsPerRevolution;
 
    enable();
    std::cout << "[Stepper] NN=" << nnOutput
              << " dir=" << (direction ? "CW" : "CCW")
              << " speed=" << speed << " µsteps/s"
              << " steps=" << stepsToTake << "\n";
 
    for (int i = 0; i < stepsToTake; ++i) {
        gpioWrite(PIN_STEP, 1);
        gpioDelayMicroseconds(stepPulseMicroseconds);
        gpioWrite(PIN_STEP, 0);
        gpioDelayMicroseconds(delayUs);
    }
}
 
void StepperMotor::driveToAngle(double targetDegrees, double speedFraction) {
    speedFraction = std::clamp(std::abs(speedFraction), 0.01, 1.0);
    double delta  = targetDegrees - currentAngleDeg;
    int usteps    = static_cast<int>(std::round((delta / 360.0) * microstepsPerRevolution));
 
    double nnEquiv = (usteps >= 0 ? 1.0 : -1.0) * speedFraction;
    driveFromNN(nnEquiv, std::abs(usteps));
    currentAngleDeg += (usteps / static_cast<double>(microstepsPerRevolution)) * 360.0;
}
 
void StepperMotor::shutdown() {
    if (running) {
        disable();
        gpioTerminate();
        running = false;
        std::cout << "[Stepper] Shutdown.\n";
    }
}
 
double StepperMotor::currentAngle() const 
            { return currentAngleDeg; }

double StepperMotor::currentSpeed() const 
            { return speed; }
 

double runNeuralNetwork(const double* inputs, int n_inputs) {
    // ── INSERT YOUR NN INFERENCE HERE ──
    double weights[] = {0.4, -0.3, 0.6};
    double sum = 0.0;
    for (int i = 0; i < std::min(n_inputs, 3); ++i)
        sum += inputs[i] * weights[i];
    return std::tanh(sum);
}
 


int main() {
    StepperMotor motor;
 
    try {
        motor.init();
 
        for (int cycle = 0; cycle < 10; ++cycle) {
            double inputs[] = {
                0.5 * std::sin(cycle * 0.5),
                0.3,
                -0.1 * cycle
            };
 
            double nnOutput = runNeuralNetwork(inputs, 3);
            std::cout << "\n[Cycle " << cycle << "] NN output: " << nnOutput << "\n";
 
            motor.driveFromNN(nnOutput, 400);
 
            std::this_thread::sleep_for(std::chrono::milliseconds(300000)); //5 minute break
        }
 
    } catch (const std::exception& e) {
        std::cerr << "[ERROR] " << e.what() << "\n";
        return 1;
    }
 
    motor.shutdown();
    return 0;
}
 