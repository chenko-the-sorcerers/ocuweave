#pragma once
#include <cstddef>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace ocuweave::nn {
using Shape = std::vector<std::size_t>;
struct Tensor;
using T = std::shared_ptr<Tensor>;
// Each operation supplies its analytic vector-Jacobian product. No external
// autograd.
struct Tensor {
  Shape shape;
  std::vector<float> data, grad;
  bool requires_grad;
  std::vector<T> parents;
  std::function<void(const std::vector<float> &)> vjp;
  Tensor(Shape shape, std::vector<float> values, bool requires_grad = false);
};
std::size_t elements(const Shape &shape);
T tensor(Shape shape, std::vector<float> values, bool requires_grad = false);
T zeros(Shape shape, bool requires_grad = false);
void backward(const T &scalar);
T add(const T &a, const T &b);
T mul(const T &a, const T &b);
T scale(const T &a, float s);
T reshape(const T &a, Shape shape);
T permute(const T &a, Shape axes);
T cat(const std::vector<T> &tensors, std::size_t axis);
T matmul(const T &a, const T &b); // BxMxK @ BxKxN; batch 1 may broadcast
T conv2d(const T &x, const T &weight, const T &bias, int stride = 1,
         int padding = 0, int groups = 1);
T group_norm(const T &x, const T &weight, const T &bias, int groups = 1,
             float eps = 1e-5f);
T gelu(const T &x);
T sigmoid(const T &x);
T softmax(const T &x, std::size_t axis);
T resize(const T &x, std::size_t height,
         std::size_t width); // align_corners=false
T reduce_sum(const T &x, std::size_t axis, bool keepdim = true);
T spatial_gate(const T &masks,
               const T &weights); // B,Q,H,W times B,Q,H,W or B,Q,1,1
struct LossResult {
  T total;
  float bce, dice;
};
LossResult observed_loss(const T &logits, const T &target, const T &valid,
                         const std::vector<unsigned char> &available,
                         const std::vector<float> &positive_weights = {});
} // namespace ocuweave::nn
