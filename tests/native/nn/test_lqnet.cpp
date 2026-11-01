#include "ocuweave/nn/lqnet.hpp"
#include <cmath>
#include <cstdio>
#include <iostream>
#include <stdexcept>
using namespace ocuweave::nn;
void check(bool b, const char *m) {
  if (!b)
    throw std::runtime_error(m);
}
int main(int argc, char **argv) {
  try {
    const std::string path = argc > 1 ? argv[1] : "native-test.ckpt";
    ModelConfig cfg;
    cfg.width = 4;
    cfg.dim = 8;
    cfg.queries = 2;
    cfg.classes = 2;
    LQNet model(cfg);
    AdamW opt(model.parameters());
    auto x = zeros({2, 3, 16, 16});
    for (std::size_t i = 0; i < x->data.size(); ++i)
      x->data[i] = .5 + .3 * std::sin(i * .03);
    auto y = zeros({2, 2, 16, 16}), valid = zeros({2, 1, 16, 16});
    std::fill(valid->data.begin(), valid->data.end(), 1);
    for (std::size_t n = 0; n < 2; ++n)
      for (int row = 4; row < 12; ++row)
        for (int col = 4; col < 12; ++col)
          y->data[(n * 2) * 256 + row * 16 + col] = 1;
    std::vector<unsigned char> available = {1, 0, 1, 1};
    auto run = [&]() {
      return observed_loss(model.forward(x), y, valid, available).total;
    };
    auto loss = run();
    float initial = loss->data[0];
    backward(loss);
    for (auto &[name, p] : model.parameters())
      for (auto g : p->grad)
        check(std::isfinite(g), "Finite model gradient");
    // Check end-to-end gradients in attention, encoder and output, away from
    // zero-gradient initialization.
    for (const auto &name :
         {"heads.0.attention.q.weight", "heads.0.mask_embedding.weight",
          "encoder.stem.0.weight", "refine.2.weight"}) {
      auto p = model.parameters().at(name);
      std::size_t i = 0;
      for (std::size_t k = 1; k < p->grad.size(); ++k)
        if (std::abs(p->grad[k]) > std::abs(p->grad[i]))
          i = k;
      float analytic = p->grad[i], old = p->data[i], eps = .002;
      p->data[i] = old + eps;
      float high = run()->data[0];
      p->data[i] = old - eps;
      float low = run()->data[0];
      p->data[i] = old;
      float numeric = (high - low) / (2 * eps);
      check(std::abs(analytic - numeric) < .008 * (1 + std::abs(numeric)),
            "End-to-end LQNet derivative");
    }
    opt.step();
    save_checkpoint(path, model, opt);
    LQNet loaded(cfg);
    AdamW loaded_opt(loaded.parameters());
    load_checkpoint(path, loaded, loaded_opt);
    check(model.forward(x)->data == loaded.forward(x)->data,
          "Checkpoint forward parity");
    backward(run());
    opt.step();
    backward(observed_loss(loaded.forward(x), y, valid, available).total);
    loaded_opt.step();
    for (auto &[name, p] : model.parameters())
      check(p->data == loaded.parameters().at(name)->data,
            "Exact resumed AdamW update");
    for (int i = 0; i < 18; ++i) {
      backward(run());
      opt.step();
    }
    check(run()->data[0] < initial, "Synthetic training loss decreases");
    model.training(false);
    check(!model.forward(x)->requires_grad,
          "Inference disables graph retention");
    for (int variant = 0; variant < 2; ++variant) {
      auto c = cfg;
      c.full_resolution_detail = variant == 0;
      c.spatial_query = variant == 1;
      LQNet alt(c);
      auto z = alt.forward(x);
      check(z->shape == y->shape, "Variant output shape");
      backward(observed_loss(z, y, valid, available).total);
    }
    std::remove(path.c_str());
    std::cout << "Native LQNet forward, gradient, training and exact "
                 "checkpoint resume passed\n";
    return 0;
  } catch (const std::exception &e) {
    std::cerr << e.what() << "\n";
    return 1;
  }
}
