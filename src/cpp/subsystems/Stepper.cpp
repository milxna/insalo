#include <subsystems/Stepper.h>

#include <cmath>
#include <stdexcept>
#include <iostream>

#ifdef __linux__
#include <pigpio.h>
#endif

Stepper::Stepper() = default;

Stepper::Stepper(const StepperConfig& config) {
    configure(config);
}

void Stepper::configure(const StepperConfig& config) {
    if (config.stepsPerRevolution <= 0)
        throw std::invalid_argument("stepsPerRevolution must be positive");
    if (config.microstepping <= 0)
        throw std::invalid_argument("microstepping must be positive");
    if (config.leadScrewPitchMm <= 0.0)
        throw std::invalid_argument("leadScrewPitchMm must be positive");
    if (config.minStepsPerSecond <= 0.0)
        throw std::invalid_argument("minStepsPerSecond must be positive");
    if (config.maxStepsPerSecond < config.minStepsPerSecond)
        throw std::invalid_argument("maxStepsPerSecond must be >= minStepsPerSecond");

    config_ = config;
}

void Stepper::init() {
#ifdef __linux__
    if (gpioInitialise() < 0)
        throw std::runtime_error("pigpio init failed — run with sudo");

    gpioSetMode(config_.pinStep, PI_OUTPUT);
    gpioSetMode(config_.pinDir,  PI_OUTPUT);
    gpioSetMode(config_.pinEn,   PI_OUTPUT);
    gpioSetMode(config_.pinMs1,  PI_OUTPUT);
    gpioSetMode(config_.pinMs2,  PI_OUTPUT);
    gpioSetMode(config_.pinMs3,  PI_OUTPUT);

    // 1/16 microstepping: MS1=MS2=MS3=HIGH
    gpioWrite(config_.pinMs1, 1);
    gpioWrite(config_.pinMs2, 1);
    gpioWrite(config_.pinMs3, 1);

    // disable motor until first move
    gpioWrite(config_.pinEn, 1);
#endif
    std::cout << "[Stepper] Initialised. " 
              << (config_.stepsPerRevolution * config_.microstepping)
              << " usteps/rev\n";
}

void Stepper::shutdown() {
#ifdef __linux__
    gpioWrite(config_.pinEn, 1);
    gpioTerminate();
#endif
    std::cout << "motor off\n";
}

void Stepper::moveSteps(int steps, double stepsPerSecond) {
    if (steps == 0) {
        lastCommandedSteps_ = 0;
        lastCommandedSpeed_ = 0.0;
        return;
    }

    const double speedMagnitude = std::abs(stepsPerSecond);
    if (speedMagnitude < config_.minStepsPerSecond ||
        speedMagnitude > config_.maxStepsPerSecond)
        throw std::out_of_range("stepsPerSecond outside configured range");

    lastCommandedSteps_ = steps;
    lastCommandedSpeed_ = stepsPerSecond;

#ifdef __linux__
    gpioWrite(config_.pinDir, steps > 0 ? 1 : 0);
    gpioDelayMicroseconds(2);
    gpioWrite(config_.pinEn, 0);

    unsigned int periodUs = static_cast<unsigned int>(1'000'000.0 / speedMagnitude);
    unsigned int delayUs  = periodUs > 5 ? periodUs - 5 : 1;

    int count = std::abs(steps);
    for (int i = 0; i < count; ++i) {
        gpioWrite(config_.pinStep, 1);
        gpioDelayMicroseconds(5);
        gpioWrite(config_.pinStep, 0);
        gpioDelayMicroseconds(delayUs);
    }

    gpioWrite(config_.pinEn, 1); //disable motor after moving
#endif

    std::cout << "stepper moved " << steps << " steps at "
              << stepsPerSecond << " steps/sec\n";
}

double Stepper::stepsPerMillimeter() const {
    return static_cast<double>(config_.stepsPerRevolution * config_.microstepping) /
           config_.leadScrewPitchMm;
}

int Stepper::lastCommandedSteps() const { return lastCommandedSteps_; }
double Stepper::lastCommandedSpeed() const { return lastCommandedSpeed_; }