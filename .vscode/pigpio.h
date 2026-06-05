//only for the sake of this being able to compile locally on my mac

#pragma once

#define PI_OUTPUT 1
#define PI_INPUT  0

inline int  gpioInitialise()
        { return 0; }
inline void gpioTerminate()
        {}
inline void gpioSetMode(int pin, int mode)
        {}
inline void gpioWrite(int pin, int level)
        {}
inline int  gpioRead(int pin)
        { return 0; }
inline void gpioDelayMicroseconds(unsigned int micros)
        {}