#pragma once
#include <cstddef>
namespace ocuweave::nn {
void set_cuda(bool enabled);
bool cuda_enabled();
bool cuda_compiled();
#ifdef OCUWEAVE_CUDA
void cuda_conv_forward(const float *, const float *, const float *, float *,
                       int, int, int, int, int, int, int, int, int, int);
void cuda_conv_backward(const float *, const float *, const float *, float *,
                        float *, float *, int, int, int, int, int, int, int,
                        int, int, int);
void cuda_matmul_forward(const float *, const float *, float *, int, int, int,
                         int, int, int);
void cuda_matmul_backward(const float *, const float *, const float *, float *,
                          float *, int, int, int, int, int, int);
#endif
} // namespace ocuweave::nn
