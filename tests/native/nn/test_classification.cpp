#include "ocuweave/nn/classification.hpp"
#include "ocuweave/nn/cuda.hpp"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <iostream>
#include <set>
#include <stdexcept>
using namespace ocuweave::nn;
void check(bool x, const char *m) {
  if (!x)
    throw std::runtime_error(m);
}
int main(int argc, char **argv) {
  try {
    bool gpu = argc > 1 && std::string(argv[1]) == "cuda";
    auto logits = tensor({2, 3}, {0, 0, 0, 1000, 999, -1000}, true);
    auto loss = cross_entropy(logits, {1, 0});
    backward(loss);
    check(std::abs(loss->data[0] -
                   (std::log(3.) + std::log1p(std::exp(-1.))) / 2) < 1e-6,
          "Analytical stable CE");
    for (std::size_t i = 0; i < logits->data.size(); ++i) {
      auto old = logits->data[i];
      float eps = .01;
      logits->data[i] = old + eps;
      auto hi = cross_entropy(logits, {1, 0})->data[0];
      logits->data[i] = old - eps;
      auto lo = cross_entropy(logits, {1, 0})->data[0];
      logits->data[i] = old;
      check(std::abs((hi - lo) / (2 * eps) - logits->grad[i]) < .002,
            "CE finite difference");
    }
    std::set<std::size_t> seen;
    for (int i = 0; i < 3; ++i)
      for (auto x : minibatch_indices(8, 3, i, 42))
        check(seen.insert(x).second, "Epoch repeats a sample");
    check(seen.size() == 8, "Epoch covers every sample");
    check(minibatch_indices(8, 3, 7, 42) == minibatch_indices(8, 3, 7, 42),
          "Resume sampler deterministic");
    auto x = zeros({4, 3, 32, 32});
    for (std::size_t i = 0; i < x->data.size(); ++i)
      x->data[i] = .5 + .3 * std::sin(i * .07);
    Classifier model(4, 4, 42);
    AdamW opt(model.parameters());
    auto initial = model.forward(x);
    auto initial_values = initial->data;
    auto initial_loss = cross_entropy(initial, {0, 1, 2, 3});
    backward(initial_loss);
    auto analytic = model.parameters().at("classifier.weight")->grad;
    auto p = model.parameters().at("classifier.weight");
    std::size_t k = std::max_element(analytic.begin(), analytic.end(),
                                     [](float a, float b) {
                                       return std::abs(a) < std::abs(b);
                                     }) -
                    analytic.begin();
    auto old = p->data[k];
    p->data[k] = old + .002;
    auto hi = cross_entropy(model.forward(x), {0, 1, 2, 3})->data[0];
    p->data[k] = old - .002;
    auto lo = cross_entropy(model.forward(x), {0, 1, 2, 3})->data[0];
    p->data[k] = old;
    check(std::abs(analytic[k] - (hi - lo) / .004) < .003,
          "Classifier end-to-end gradient");
    if (gpu) {
      Classifier other(4, 4, 42);
      set_cuda(true);
      auto z = other.forward(x);
      backward(cross_entropy(z, {0, 1, 2, 3}));
      for (std::size_t i = 0; i < z->data.size(); ++i)
        check(std::abs(z->data[i] - initial_values[i]) < 2e-4,
              "CUDA classifier logits");
      for (auto &[name, param] : model.parameters())
        for (std::size_t i = 0; i < param->grad.size(); ++i)
          check(
              std::abs(param->grad[i] - other.parameters().at(name)->grad[i]) <
                  3e-4 * (1 + std::abs(param->grad[i])),
              "CUDA classifier gradient");
      set_cuda(false);
    }
    const std::string path =
        gpu ? "classifier-cuda.ckpt" : "classifier-cpu.ckpt";
    opt.step();
    save_checkpoint(path, model.state(), opt);
    Classifier restored(4, 4, 42);
    AdamW restored_opt(restored.parameters());
    load_checkpoint(path, restored.state(), restored_opt);
    for (auto *m : {&model, &restored})
      backward(cross_entropy(m->forward(x), {0, 1, 2, 3}));
    opt.step();
    restored_opt.step();
    for (auto &[name, param] : model.parameters())
      check(param->data == restored.parameters().at(name)->data,
            "Exact classifier resume");
    for (int i = 0; i < 30; ++i) {
      backward(cross_entropy(model.forward(x), {0, 1, 2, 3}));
      opt.step();
    }
    check(cross_entropy(model.forward(x), {0, 1, 2, 3})->data[0] <
              initial_loss->data[0],
          "Classifier learns synthetic fixture");
    model.training(false);
    check(!model.forward(x)->requires_grad, "Classifier inference no graph");
    std::remove(path.c_str());
    std::cout << "Classification CE, sampler, gradients, learning and resume "
                 "passed\n";
    return 0;
  } catch (const std::exception &e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
