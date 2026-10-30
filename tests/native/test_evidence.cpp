#include "ocuweave/evidence.hpp"
#include <cmath>
#include <limits>
#include <stdexcept>
int main() {
    double values[] = {2, 8, std::numeric_limits<double>::quiet_NaN()};
    double weights[] = {1, 3, 1};
    unsigned char present[] = {1, 1, 0};
    if (std::abs(ocuweave::masked_weighted_mean(values, weights, present, 3) - 6.5) > 1e-12) return 1;
    present[0] = present[1] = 0;
    try { ocuweave::masked_weighted_mean(values, weights, present, 3); }
    catch (const std::invalid_argument&) { return 0; }
    return 2;
}
