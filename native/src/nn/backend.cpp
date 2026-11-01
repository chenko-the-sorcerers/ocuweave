#include "ocuweave/nn/cuda.hpp"
#include <stdexcept>
namespace ocuweave::nn {
namespace {
bool enabled = false;
}
bool cuda_compiled() {
#ifdef OCUWEAVE_CUDA
  return true;
#else
  return false;
#endif
}
bool cuda_enabled() { return enabled; }
void set_cuda(bool value) {
  if (value && !cuda_compiled())
    throw std::runtime_error(
        "CUDA backend not compiled; configure -DOCUWEAVE_ENABLE_CUDA=ON");
  enabled = value;
}
} // namespace ocuweave::nn
