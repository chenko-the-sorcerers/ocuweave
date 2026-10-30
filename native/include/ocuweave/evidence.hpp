#pragma once
#include <cstddef>
namespace ocuweave {
// CPU baseline only. Missing entries are ignored, including their stored values.
double masked_weighted_mean(const double* values, const double* weights,
                           const unsigned char* available, std::size_t count);
}
