#include "ocuweave/nn/cuda.hpp"
#include "ocuweave/nn/lqnet.hpp"
#include <cmath>
#include <cstdio>
#include <functional>
#include <iostream>
#include <stdexcept>
using namespace ocuweave::nn;
void check(bool b, const char *m) {
  if (!b)
    throw std::runtime_error(m);
}
void near(double a, double b, double tol, const char *m) {
  if (std::abs(a - b) > tol) {
    std::cerr << m << ": " << a << " versus " << b << "\n";
    throw std::runtime_error(m);
  }
}
T values(Shape s, float shift = 0) {
  auto x = zeros(s, true);
  for (std::size_t i = 0; i < x->data.size(); ++i)
    x->data[i] = shift + .4 * std::sin(i * .71 + .2);
  return x;
}
T objective(T x) {
  auto w = zeros(x->shape);
  for (std::size_t i = 0; i < w->data.size(); ++i)
    w->data[i] = .2 + std::cos(i * .57);
  x = mul(x, w);
  while (x->shape.size() > 1)
    x = reduce_sum(x, x->shape.size() - 1, false);
  return reduce_sum(x, 0);
}
void gradient(const std::string &name, const std::function<T()> &fn,
              const std::vector<T> &inputs, float tolerance = .003f) {
  auto out = fn();
  backward(objective(out));
  std::vector<std::vector<float>> expected;
  for (auto &x : inputs)
    expected.push_back(x->grad);
  for (std::size_t j = 0; j < inputs.size(); ++j) {
    auto x = inputs[j];
    for (std::size_t i = 0; i < x->data.size();
         i += std::max(std::size_t(1), x->data.size() / 23)) {
      float before = x->data[i], epsilon = .002f;
      x->data[i] = before + epsilon;
      float hi = objective(fn())->data[0];
      x->data[i] = before - epsilon;
      float lo = objective(fn())->data[0];
      x->data[i] = before;
      double numeric = (hi - lo) / (2 * epsilon);
      near(expected[j][i], numeric, tolerance * (1 + std::abs(numeric)),
           name.c_str());
    }
  }
  std::cout << "gradient " << name << " passed\n";
}
int main(int argc, char **argv) {
  if (argc > 1 && std::string(argv[1]) == "cuda")
    set_cuda(true);
  try {
    auto a = values({2, 3, 2, 3}), b = values({1, 3, 1, 1});
    gradient("broadcast", [&] { return mul(add(a, b), b); }, {a, b});
    gradient("permute/reshape",
             [&] { return reshape(permute(a, {0, 2, 3, 1}), {2, 6, 3}); }, {a});
    gradient("cat/reduce", [&] { return reduce_sum(cat({a, a}, 1), 2, false); },
             {a});
    auto ma = values({2, 3, 4}), mb = values({1, 4, 2});
    gradient("batched matrix product", [&] { return matmul(ma, mb); },
             {ma, mb});
    auto x = values({2, 4, 5, 6}), w = values({6, 2, 3, 2}), bias = values({6});
    gradient(
        "grouped convolution", [&] { return conv2d(x, w, bias, 2, 1, 2); },
        {x, w, bias}, .006);
    auto nw = values({4}, 1), nb = values({4});
    gradient(
        "group normalization", [&] { return group_norm(x, nw, nb, 2); },
        {x, nw, nb}, .008);
    gradient("GELU", [&] { return gelu(a); }, {a});
    gradient("sigmoid", [&] { return sigmoid(a); }, {a});
    gradient("softmax", [&] { return softmax(a, 1); }, {a});
    gradient("bilinear resize", [&] { return resize(a, 5, 7); }, {a}, .008);
    auto logits = values({2, 2, 2, 3}), target = zeros(logits->shape),
         valid = zeros({2, 1, 2, 3});
    std::fill(valid->data.begin(), valid->data.end(), 1);
    valid->data[1] = 0;
    target->data[0] = 1;
    target->data[15] = 1;
    target->data[18] = 1;
    std::vector<unsigned char> available = {1, 0, 1, 1};
    gradient(
        "weighted observed BCE + Dice",
        [&] {
          return observed_loss(logits, target, valid, available, {3, 2}).total;
        },
        {logits});
    backward(observed_loss(logits, target, valid, available, {3, 2}).total);
    near(logits->grad[1], 0, 0, "padding excluded");
    near(logits->grad[6], 0, 0, "missing label excluded");
    logits->data[6] = NAN;
    check(std::isfinite(
              observed_loss(logits, target, valid, available).total->data[0]),
          "Missing NaN ignored");
    auto zz = tensor({1, 1, 1, 2}, {0, 0}, true),
         yy = tensor({1, 1, 1, 2}, {1, 0}), vv = tensor({1, 1, 1, 2}, {1, 1});
    auto loss = observed_loss(zz, yy, vv, {1}, {3});
    near(loss.bce, 2 * std::log(2.), 1e-6, "weighted BCE oracle");
    near(loss.dice, 1. / 3, 1e-6, "Dice oracle");
    backward(loss.total);
    near(zz->grad[0], -.75 - 1. / 9, 1e-6, "positive gradient oracle");
    near(zz->grad[1], .25 + 1. / 18, 1e-6, "negative gradient oracle");
    std::map<std::string, T> params = {{"p", tensor({1}, {1}, true)}};
    AdamWConfig ac;
    ac.learning_rate = .01;
    ac.weight_decay = .1;
    ac.clip = 100;
    AdamW adam(params, ac);
    params["p"]->grad[0] = 2;
    adam.step();
    near(params["p"]->data[0], .989, 1e-6, "AdamW first step oracle");
    std::cout << "All native operator checks passed\n";
    return 0;
  } catch (const std::exception &e) {
    std::cerr << e.what() << "\n";
    return 1;
  }
}
