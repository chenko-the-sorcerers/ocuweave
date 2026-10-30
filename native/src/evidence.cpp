#include "ocuweave/evidence.hpp"
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace ocuweave {
double masked_weighted_mean(const double* values, const double* weights,
                           const unsigned char* available, std::size_t count) {
    if (!values || !weights || !available || count == 0)
        throw std::invalid_argument("empty inputs");
    double scale = 0;
    for (std::size_t i = 0; i < count; ++i) {
        if (available[i] > 1) throw std::invalid_argument("invalid availability");
        if (!available[i]) continue;
        if (!std::isfinite(values[i]) || !std::isfinite(weights[i]) || weights[i] < 0)
            throw std::invalid_argument("invalid available evidence");
        scale = std::max(scale, weights[i]);
    }
    if (scale <= 0) throw std::invalid_argument("no positive-weight evidence");
    long double denominator = 0, result = 0;
    for (std::size_t i = 0; i < count; ++i)
        if (available[i]) denominator += static_cast<long double>(weights[i]) / scale;
    for (std::size_t i = 0; i < count; ++i)
        if (available[i]) result += values[i] * ((static_cast<long double>(weights[i]) / scale) / denominator);
    const double output = static_cast<double>(result);
    if (!std::isfinite(output)) throw std::invalid_argument("non-finite result");
    return output;
}
}
