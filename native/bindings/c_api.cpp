#include "ocuweave/evidence.hpp"
#ifdef _WIN32
#define OW_EXPORT __declspec(dllexport)
#else
#define OW_EXPORT __attribute__((visibility("default")))
#endif
// Never let a C++ exception cross the C ABI boundary.
extern "C" OW_EXPORT int ow_masked_weighted_mean(
    const double* values, const double* weights, const unsigned char* available,
    std::size_t count, double* output) noexcept {
    if (!output) return 1;
    try {
        *output = ocuweave::masked_weighted_mean(values, weights, available, count);
        return 0;
    } catch (...) { return 1; }
}
