#pragma once

struct milliliter_t {
    double value();
};

struct millimeter_t {
    double value();
};

struct second_t {
    double value();
};

struct milliliters_per_second_t {
    double value();
};

namespace units::literal {

    constexpr milliliter_t operator"" _mL(long double value) {
        return milliliter_t{static_cast<double>(value)};
    }

        constexpr milliliter_t operator"" _mL(unsigned long long value) {
            return milliliter_t{static_cast<double>(value)};
        }

    constexpr second_t operator"" _s(long double value) {
        return second_t{static_cast<double>(value)};
    }

        constexpr second_t operator"" _s(unsigned long long value) {
            return second_t{static_cast<double>(value)};
        }

    constexpr millimeter_t operator"" _mm(long double value) {
        return millimeter_t{static_cast<double>(value)};
    }

        constexpr millimeter_t operator"" _mm(unsigned long long value) {
            return millimeter_t{static_cast<double>(value)};
        }

    constexpr milliliters_per_second_t operator/(milliliter_t volume,
                                             second_t time)
    {
        return milliliters_per_second_t{
            volume.value() / time.value()
        };
    }

}
