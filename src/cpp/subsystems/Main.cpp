#include <iostream>
#include <stdexcept>
#include <Stepper.h>


//test code 
int main() {
    try {
        motor.init();

        std::cout << "Moving forward one full revolution...\n";
        motor.moveSteps(3200, 800.0);   

        std::cout << "Moving backward one full revolution...\n";
        motor.moveSteps(-3200, 800.0);

        std::cout << "Steps per mm: " << motor.stepsPerMillimeter() << "\n";

    } catch (const std::exception& e) {
        std::cerr << "error " << e.what() << "\n";
        motor.shutdown();
        return 1;
    }

    motor.shutdown();
    return 0;
}