#include "ocuweave/nn/cuda.hpp"
#include "ocuweave/nn/lqnet.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>
using namespace ocuweave::nn;
void compare(const std::vector<float> &a, const std::vector<float> &b,
             float tolerance, const char *what) {
  if (a.size() != b.size())
    throw std::runtime_error("Shape mismatch");
  float worst = 0;
  for (std::size_t i = 0; i < a.size(); ++i) {
    if (!std::isfinite(a[i]) || !std::isfinite(b[i]))
      throw std::runtime_error("Nonfinite parity value");
    worst = std::max(worst, std::abs(a[i] - b[i]) / (1 + std::abs(a[i])));
  }
  std::cout << what << " max scaled error " << worst << '\n';
  if (worst > tolerance)
    throw std::runtime_error(what);
}
int main() {
  try {
    ModelConfig c;
    c.width = 4;
    c.dim = 8;
    c.queries = 2;
    c.classes = 2;
    LQNet cpu(c), gpu(c);
    auto x = zeros({2, 3, 32, 32}), y = zeros({2, 2, 32, 32}),
         v = zeros({2, 1, 32, 32});
    for (std::size_t i = 0; i < x->data.size(); ++i)
      x->data[i] = .4 + .3 * std::sin(i * .02);
    for (std::size_t i = 0; i < y->data.size(); ++i)
      y->data[i] = i % 17 == 0;
    std::fill(v->data.begin(), v->data.end(), 1);
    set_cuda(false);
    auto a = cpu.forward(x);
    backward(observed_loss(a, y, v, {1, 0, 1, 1}, {2, 3}).total);
    set_cuda(true);
    auto b = gpu.forward(x);
    backward(observed_loss(b, y, v, {1, 0, 1, 1}, {2, 3}).total);
    compare(a->data, b->data, 2e-4, "CUDA forward");
    for (auto &[name, p] : cpu.parameters())
      compare(p->grad, gpu.parameters().at(name)->grad, 3e-4, name.c_str());
    AdamW ao(cpu.parameters()), bo(gpu.parameters());
    ao.step();
    bo.step();
    for (auto &[name, p] : cpu.parameters())
      compare(p->data, gpu.parameters().at(name)->data, 3e-4, name.c_str());
    std::cout << "CUDA parity passed\n";
    return 0;
  } catch (const std::exception &e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
