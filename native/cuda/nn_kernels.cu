// Dependency boundary: CUDA runtime only; no cuDNN, cuBLAS, LibTorch or other
// ML framework.
#include "ocuweave/nn/cuda.hpp"
#include <algorithm>
#include <cuda_runtime.h>
#include <stdexcept>
#include <string>
namespace ocuweave::nn {
namespace {
void checked(cudaError_t e) {
  if (e != cudaSuccess)
    throw std::runtime_error(std::string("CUDA: ") + cudaGetErrorString(e));
}
struct Buffer {
  float *p = nullptr;
  std::size_t n;
  explicit Buffer(std::size_t size, const float *source = nullptr) : n(size) {
    checked(cudaMalloc(&p, n * sizeof(float)));
    try {
      if (source)
        checked(
            cudaMemcpy(p, source, n * sizeof(float), cudaMemcpyHostToDevice));
      else
        checked(cudaMemset(p, 0, n * sizeof(float)));
    } catch (...) {
      cudaFree(p);
      throw;
    }
  }
  ~Buffer() { cudaFree(p); }
  Buffer(const Buffer &) = delete;
  Buffer &operator=(const Buffer &) = delete;
  void copy(float *out) {
    checked(cudaMemcpy(out, p, n * sizeof(float), cudaMemcpyDeviceToHost));
  }
};
__global__ void conv_f(const float *x, const float *w, const float *b, float *y,
                       int N, int C, int H, int W, int O, int KH, int KW, int S,
                       int P, int G, int OH, int OW) {
  int k = blockIdx.x * blockDim.x + threadIdx.x;
  if (k >= N * O * OH * OW)
    return;
  int ox = k % OW, oy = k / OW % OH, oc = k / (OW * OH) % O,
      n = k / (OW * OH * O);
  float v = b[oc];
  for (int ci = 0; ci < C / G; ++ci)
    for (int ky = 0; ky < KH; ++ky)
      for (int kx = 0; kx < KW; ++kx) {
        int iy = oy * S + ky - P, ix = ox * S + kx - P;
        if (iy < 0 || iy >= H || ix < 0 || ix >= W)
          continue;
        int ic = oc / (O / G) * (C / G) + ci;
        v += x[((n * C + ic) * H + iy) * W + ix] *
             w[((oc * (C / G) + ci) * KH + ky) * KW + kx];
      }
  y[k] = v;
}
__global__ void conv_dx(const float *w, const float *g, float *dx, int N, int C,
                        int H, int W, int O, int KH, int KW, int S, int P,
                        int G, int OH, int OW) {
  int k = blockIdx.x * blockDim.x + threadIdx.x;
  if (k >= N * C * H * W)
    return;
  int ix = k % W, iy = k / W % H, ic = k / (W * H) % C, n = k / (W * H * C),
      group = ic / (C / G);
  float v = 0;
  for (int oc = group * (O / G); oc < (group + 1) * (O / G); ++oc)
    for (int ky = 0; ky < KH; ++ky)
      for (int kx = 0; kx < KW; ++kx) {
        int yy = iy + P - ky, xx = ix + P - kx;
        if (yy < 0 || xx < 0 || yy % S || xx % S)
          continue;
        int oy = yy / S, ox = xx / S;
        if (oy >= OH || ox >= OW)
          continue;
        v += g[((n * O + oc) * OH + oy) * OW + ox] *
             w[((oc * (C / G) + ic % (C / G)) * KH + ky) * KW + kx];
      }
  dx[k] = v;
}
__global__ void conv_dw(const float *x, const float *g, float *dw, int N, int C,
                        int H, int W, int O, int KH, int KW, int S, int P,
                        int G, int OH, int OW) {
  int k = blockIdx.x, kx = k % KW, ky = k / KW % KH,
      ci = k / (KW * KH) % (C / G), oc = k / (KW * KH * (C / G)),
      ic = oc / (O / G) * (C / G) + ci;
  float v = 0;
  for (int i = threadIdx.x; i < N * OH * OW; i += blockDim.x) {
    int ox = i % OW, oy = i / OW % OH, n = i / (OW * OH), iy = oy * S + ky - P,
        ix = ox * S + kx - P;
    if (iy >= 0 && iy < H && ix >= 0 && ix < W)
      v += x[((n * C + ic) * H + iy) * W + ix] *
           g[((n * O + oc) * OH + oy) * OW + ox];
  }
  __shared__ float sum[256];
  sum[threadIdx.x] = v;
  __syncthreads();
  for (int d = 128; d; d /= 2) {
    if (threadIdx.x < d)
      sum[threadIdx.x] += sum[threadIdx.x + d];
    __syncthreads();
  }
  if (threadIdx.x == 0)
    dw[k] = sum[0];
}
__global__ void conv_db(const float *g, float *db, int N, int O, int S) {
  int c = blockIdx.x;
  float v = 0;
  for (int i = threadIdx.x; i < N * S; i += blockDim.x)
    v += g[(i / S * O + c) * S + i % S];
  __shared__ float sum[256];
  sum[threadIdx.x] = v;
  __syncthreads();
  for (int d = 128; d; d /= 2) {
    if (threadIdx.x < d)
      sum[threadIdx.x] += sum[threadIdx.x + d];
    __syncthreads();
  }
  if (threadIdx.x == 0)
    db[c] = sum[0];
}
__global__ void mm_f(const float *a, const float *b, float *c, int B, int A,
                     int BB, int M, int K, int N) {
  int j = blockIdx.x * 16 + threadIdx.x, i = blockIdx.y * 16 + threadIdx.y,
      q = blockIdx.z;
  __shared__ float aa[16][16], bb[16][16];
  float v = 0;
  for (int k = 0; k < K; k += 16) {
    aa[threadIdx.y][threadIdx.x] =
        (i < M && k + threadIdx.x < K)
            ? a[(q % A * M + i) * K + k + threadIdx.x]
            : 0;
    bb[threadIdx.y][threadIdx.x] =
        (j < N && k + threadIdx.y < K)
            ? b[(q % BB * K + k + threadIdx.y) * N + j]
            : 0;
    __syncthreads();
    for (int t = 0; t < 16; ++t)
      v += aa[threadIdx.y][t] * bb[t][threadIdx.x];
    __syncthreads();
  }
  if (i < M && j < N)
    c[(q * M + i) * N + j] = v;
}
__global__ void mm_da(const float *b, const float *g, float *da, int B, int A,
                      int BB, int M, int K, int N) {
  int z = blockIdx.x * blockDim.x + threadIdx.x;
  if (z >= A * M * K)
    return;
  int k = z % K, i = z / K % M, ba = z / (K * M);
  float v = 0;
  for (int q = ba; q < B; q += A)
    for (int j = 0; j < N; ++j)
      v += g[(q * M + i) * N + j] * b[(q % BB * K + k) * N + j];
  da[z] = v;
}
__global__ void mm_db(const float *a, const float *g, float *db, int B, int A,
                      int BB, int M, int K, int N) {
  int z = blockIdx.x * blockDim.x + threadIdx.x;
  if (z >= BB * K * N)
    return;
  int j = z % N, k = z / N % K, bb = z / (N * K);
  float v = 0;
  for (int q = bb; q < B; q += BB)
    for (int i = 0; i < M; ++i)
      v += a[(q % A * M + i) * K + k] * g[(q * M + i) * N + j];
  db[z] = v;
}
} // namespace
void cuda_conv_forward(const float *x, const float *w, const float *b,
                       float *out, int N, int C, int H, int W, int O, int KH,
                       int KW, int S, int P, int G) {
  int OH = (H + 2 * P - KH) / S + 1, OW = (W + 2 * P - KW) / S + 1;
  Buffer xx(std::size_t(N) * C * H * W, x),
      ww(std::size_t(O) * C / G * KH * KW, w), bb(O, b),
      yy(std::size_t(N) * O * OH * OW);
  conv_f<<<(yy.n + 255) / 256, 256>>>(xx.p, ww.p, bb.p, yy.p, N, C, H, W, O, KH,
                                      KW, S, P, G, OH, OW);
  checked(cudaGetLastError());
  yy.copy(out);
}
void cuda_conv_backward(const float *x, const float *w, const float *g,
                        float *dx, float *dw, float *db, int N, int C, int H,
                        int W, int O, int KH, int KW, int S, int P, int G) {
  int OH = (H + 2 * P - KH) / S + 1, OW = (W + 2 * P - KW) / S + 1;
  Buffer xx(std::size_t(N) * C * H * W, x),
      ww(std::size_t(O) * C / G * KH * KW, w),
      gg(std::size_t(N) * O * OH * OW, g), gx(xx.n), gw(ww.n), gb(O);
  conv_dx<<<(gx.n + 255) / 256, 256>>>(ww.p, gg.p, gx.p, N, C, H, W, O, KH, KW,
                                       S, P, G, OH, OW);
  conv_dw<<<gw.n, 256>>>(xx.p, gg.p, gw.p, N, C, H, W, O, KH, KW, S, P, G, OH,
                         OW);
  conv_db<<<O, 256>>>(gg.p, gb.p, N, O, OH * OW);
  checked(cudaGetLastError());
  gx.copy(dx);
  gw.copy(dw);
  gb.copy(db);
}
void cuda_matmul_forward(const float *a, const float *b, float *c, int B, int A,
                         int BB, int M, int K, int N) {
  Buffer aa(std::size_t(A) * M * K, a), bb(std::size_t(BB) * K * N, b),
      cc(std::size_t(B) * M * N);
  mm_f<<<dim3((N + 15) / 16, (M + 15) / 16, B), dim3(16, 16)>>>(
      aa.p, bb.p, cc.p, B, A, BB, M, K, N);
  checked(cudaGetLastError());
  cc.copy(c);
}
void cuda_matmul_backward(const float *a, const float *b, const float *g,
                          float *da, float *db, int B, int A, int BB, int M,
                          int K, int N) {
  Buffer aa(std::size_t(A) * M * K, a), bb(std::size_t(BB) * K * N, b),
      gg(std::size_t(B) * M * N, g), ga(aa.n), gb(bb.n);
  mm_da<<<(ga.n + 255) / 256, 256>>>(bb.p, gg.p, ga.p, B, A, BB, M, K, N);
  mm_db<<<(gb.n + 255) / 256, 256>>>(aa.p, gg.p, gb.p, B, A, BB, M, K, N);
  checked(cudaGetLastError());
  ga.copy(da);
  gb.copy(db);
}
} // namespace ocuweave::nn
