#include "ocuweave/nn/lqnet.hpp"
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <limits>
#include <stdexcept>

namespace ocuweave::nn {
namespace {
std::uint32_t name_seed(const std::string &s, std::uint32_t seed) {
  std::uint32_t h = 2166136261u ^ seed;
  for (unsigned char c : s) {
    h ^= c;
    h *= 16777619u;
  }
  return h ? h : 1;
}
float uniform(std::uint32_t &state) {
  state ^= state << 13;
  state ^= state >> 17;
  state ^= state << 5;
  return (double(state) + .5) / 4294967296.;
}
void ensure(bool b, const char *m) {
  if (!b)
    throw std::invalid_argument(m);
}
template <class V> void write(std::ostream &s, const V &v) {
  s.write(reinterpret_cast<const char *>(&v), sizeof(v));
}
template <class V> void read(std::istream &s, V &v) {
  s.read(reinterpret_cast<char *>(&v), sizeof(v));
  if (!s)
    throw std::runtime_error("Truncated checkpoint");
}
void vector_write(std::ostream &s, const std::vector<float> &v) {
  s.write(reinterpret_cast<const char *>(v.data()), v.size() * sizeof(float));
}
void vector_read(std::istream &s, std::vector<float> &v) {
  s.read(reinterpret_cast<char *>(v.data()), v.size() * sizeof(float));
  if (!s)
    throw std::runtime_error("Truncated checkpoint tensor");
  for (auto x : v)
    ensure(std::isfinite(x), "Nonfinite checkpoint tensor");
}
} // namespace
T LQNet::parameter(const std::string &name, Shape shape, float bound, bool zero,
                   bool normal) {
  auto p = zeros(std::move(shape), true);
  auto state = name_seed(name, config_.seed);
  if (!zero)
    for (auto &v : p->data) {
      if (normal) {
        double u = uniform(state), q = uniform(state);
        v = bound * std::sqrt(-2 * std::log(u)) *
            std::cos(6.283185307179586 * q);
      } else
        v = (2 * uniform(state) - 1) * bound;
    }
  parameters_.emplace(name, p);
  return p;
}
void LQNet::convolution(const std::string &p, int in, int out, int k,
                        int groups, bool zero) {
  float bound = 1 / std::sqrt(float(in / groups * k * k));
  parameter(p + ".weight",
            {std::size_t(out), std::size_t(in / groups), std::size_t(k),
             std::size_t(k)},
            bound, zero);
  parameter(p + ".bias", {std::size_t(out)}, bound, zero);
}
void LQNet::norm_parameters(const std::string &p, int c) {
  auto w = parameter(p + ".weight", {std::size_t(c)}, 0, true);
  std::fill(w->data.begin(), w->data.end(), 1);
  parameter(p + ".bias", {std::size_t(c)}, 0, true);
}
void LQNet::linear_parameters(const std::string &p, int in, int out) {
  float bound = 1 / std::sqrt(float(in));
  parameter(p + ".weight", {std::size_t(out), std::size_t(in)}, bound);
  parameter(p + ".bias", {std::size_t(out)}, bound);
}
void LQNet::mixer_parameters(const std::string &p, int c) {
  convolution(p + ".spatial", c, c, 7, c);
  norm_parameters(p + ".norm", c);
  convolution(p + ".channel.0", c, 2 * c, 1);
  convolution(p + ".channel.2", 2 * c, c, 1);
  auto s = parameter(p + ".scale", {1, std::size_t(c), 1, 1}, 0, true);
  std::fill(s->data.begin(), s->data.end(), .01f);
}
LQNet::LQNet(ModelConfig cfg) : config_(cfg) {
  ensure(cfg.width >= 4 && cfg.width <= 256 && cfg.dim > 0 && cfg.dim <= 512 &&
             cfg.dim % 4 == 0 && cfg.queries > 0 && cfg.queries <= 128 &&
             cfg.classes > 0 && cfg.classes <= 32,
         "Invalid LQNet configuration");
  ensure(cfg.local_detail || !cfg.full_resolution_detail,
         "Full-resolution detail requires local branch");
  int w = cfg.width, d = cfg.dim;
  convolution("encoder.stem.0", 3, w, 4);
  norm_parameters("encoder.stem.1", w);
  mixer_parameters("encoder.stages.0", w);
  convolution("encoder.stages.1.0", w, 2 * w, 2);
  mixer_parameters("encoder.stages.1.1", 2 * w);
  convolution("encoder.stages.2.0", 2 * w, 4 * w, 2);
  mixer_parameters("encoder.stages.2.1", 4 * w);
  for (int c = 0; c < cfg.classes; ++c) {
    auto p =
        cfg.classes == 1 ? std::string("head") : "heads." + std::to_string(c);
    for (int i = 0; i < 3; ++i)
      convolution(p + ".pixels.projections." + std::to_string(i), w * (1 << i),
                  d, 1);
    convolution(p + ".pixels.mix.0", 3 * d, d, 1);
    norm_parameters(p + ".pixels.mix.1", d);
    parameter(p + ".query", {1, std::size_t(cfg.queries), std::size_t(d)},
              1 / std::sqrt(float(d)), false, true);
    for (auto name : {"q", "k", "v", "out"})
      linear_parameters(p + ".attention." + name, d, d);
    // Equivalent Q/K/V decomposition; attention biases begin at zero.
    for (auto name : {"q", "k", "v", "out"})
      std::fill(
          parameters_.at(p + ".attention." + name + ".bias")->data.begin(),
          parameters_.at(p + ".attention." + name + ".bias")->data.end(), 0);
    norm_parameters(p + ".norm", d);
    linear_parameters(p + ".mask_embedding", d, d);
    linear_parameters(p + ".query_weight", d, 1);
    parameter(p + ".bias", {1}, 0, true)->data[0] = -2;
    if (cfg.spatial_query)
      convolution(p + ".spatial_router", d, cfg.queries, 1, 1, true);
  }
  if (cfg.local_detail) {
    convolution("detail.0", 3, w, 3);
    norm_parameters("detail.1", w);
    convolution("detail.3", w, w, 3, w);
    norm_parameters("detail.4", w);
    convolution("refine.0", w + cfg.classes, w, 1);
    convolution("refine.2", w, cfg.classes, 1, 1, true);
  }
}
T LQNet::convolution_forward(const std::string &p, const T &x, int stride,
                             int padding, int groups) {
  return conv2d(x, parameters_.at(p + ".weight"), parameters_.at(p + ".bias"),
                stride, padding, groups);
}
T LQNet::norm_forward(const std::string &p, const T &x) {
  return group_norm(x, parameters_.at(p + ".weight"),
                    parameters_.at(p + ".bias"));
}
T LQNet::linear_forward(const std::string &p, const T &x) {
  auto w = parameters_.at(p + ".weight");
  return add(
      matmul(x, reshape(permute(w, {1, 0}), {1, w->shape[1], w->shape[0]})),
      reshape(parameters_.at(p + ".bias"), {1, 1, w->shape[0]}));
}
T LQNet::mixer_forward(const std::string &p, const T &x) {
  auto v = convolution_forward(p + ".spatial", x, 1, 3, x->shape[1]);
  v = norm_forward(p + ".norm", v);
  v = gelu(convolution_forward(p + ".channel.0", v));
  v = convolution_forward(p + ".channel.2", v);
  return add(x, mul(parameters_.at(p + ".scale"), v));
}
T LQNet::head_forward(const std::string &p, const std::vector<T> &f,
                      std::size_t H, std::size_t W) {
  std::vector<T> projected;
  for (int i = 0; i < 3; ++i)
    projected.push_back(convolution_forward(
        p + ".pixels.projections." + std::to_string(i), f[i]));
  auto B = f[0]->shape[0], D = std::size_t(config_.dim),
       Q = std::size_t(config_.queries), h = f[0]->shape[2], w = f[0]->shape[3],
       S = projected[2]->shape[2] * projected[2]->shape[3];
  auto context = permute(reshape(projected[2], {B, D, S}), {0, 2, 1});
  auto pixels = gelu(norm_forward(
      p + ".pixels.mix.1",
      convolution_forward(p + ".pixels.mix.0",
                          cat({projected[0], resize(projected[1], h, w),
                               resize(projected[2], h, w)},
                              1))));
  auto query = add(parameters_.at(p + ".query"), zeros({B, Q, D}));
  auto heads = [&](T x, std::size_t n) {
    return reshape(permute(reshape(x, {B, n, 4, D / 4}), {0, 2, 1, 3}),
                   {B * 4, n, D / 4});
  };
  auto q = heads(linear_forward(p + ".attention.q", query), Q),
       k = heads(linear_forward(p + ".attention.k", context), S),
       v = heads(linear_forward(p + ".attention.v", context), S);
  auto attended = matmul(softmax(scale(matmul(q, permute(k, {0, 2, 1})),
                                       1 / std::sqrt(float(D / 4))),
                                 2),
                         v);
  attended = reshape(permute(reshape(attended, {B, 4, Q, D / 4}), {0, 2, 1, 3}),
                     {B, Q, D});
  auto residual = add(query, linear_forward(p + ".attention.out", attended));
  query =
      reshape(norm_forward(p + ".norm", reshape(residual, {B * Q, D, 1, 1})),
              {B, Q, D});
  auto masks =
      reshape(scale(matmul(linear_forward(p + ".mask_embedding", query),
                           reshape(pixels, {B, D, h * w})),
                    1 / std::sqrt(float(D))),
              {B, Q, h, w});
  auto scores =
      reshape(linear_forward(p + ".query_weight", query), {B, Q, 1, 1});
  if (config_.spatial_query)
    scores = add(scores, convolution_forward(p + ".spatial_router", pixels));
  auto logits =
      add(spatial_gate(masks, softmax(scores, 1)), parameters_.at(p + ".bias"));
  return resize(logits, H, W);
}
T LQNet::forward(const T &x) {
  ensure(x->shape.size() == 4 && x->shape[1] == 3 && x->shape[2] >= 16 &&
             x->shape[3] >= 16,
         "LQNet expects N,3,H,W with H/W >=16");
  std::vector<T> f;
  auto z = gelu(norm_forward("encoder.stem.1",
                             convolution_forward("encoder.stem.0", x, 4)));
  z = mixer_forward("encoder.stages.0", z);
  f.push_back(z);
  z = mixer_forward("encoder.stages.1.1",
                    convolution_forward("encoder.stages.1.0", z, 2));
  f.push_back(z);
  z = mixer_forward("encoder.stages.2.1",
                    convolution_forward("encoder.stages.2.0", z, 2));
  f.push_back(z);
  std::vector<T> masks;
  for (int c = 0; c < config_.classes; ++c)
    masks.push_back(head_forward(
        config_.classes == 1 ? "head" : "heads." + std::to_string(c), f,
        x->shape[2], x->shape[3]));
  auto logits = cat(masks, 1);
  if (!config_.local_detail)
    return logits;
  auto local = gelu(norm_forward(
      "detail.1",
      convolution_forward("detail.0", x, config_.full_resolution_detail ? 1 : 2,
                          1)));
  local = gelu(norm_forward(
      "detail.4", convolution_forward("detail.3", local, 1, 1, config_.width)));
  auto residual = convolution_forward(
      "refine.2",
      gelu(convolution_forward(
          "refine.0",
          cat({local, resize(logits, local->shape[2], local->shape[3])}, 1))));
  return add(logits, resize(residual, x->shape[2], x->shape[3]));
}
void LQNet::training(bool enabled) {
  for (auto &[name, p] : parameters_) {
    p->requires_grad = enabled;
    if (enabled && p->grad.size() != p->data.size())
      p->grad.assign(p->data.size(), 0);
  }
}
std::size_t LQNet::parameter_count() const {
  std::size_t n = 0;
  for (auto &[name, p] : parameters_)
    n += p->data.size();
  return n;
}
AdamW::AdamW(std::map<std::string, T> &p, AdamWConfig c)
    : config(c), parameters_(p) {
  ensure(std::isfinite(c.learning_rate) && c.learning_rate > 0 &&
             std::isfinite(c.weight_decay) && c.weight_decay >= 0 &&
             c.beta1 >= 0 && c.beta1 < 1 && c.beta2 >= 0 && c.beta2 < 1 &&
             c.epsilon > 0 && std::isfinite(c.epsilon) && c.clip > 0 &&
             std::isfinite(c.clip),
         "Invalid optimizer configuration");
  for (auto &[name, t] : p) {
    first[name].assign(t->data.size(), 0);
    second[name].assign(t->data.size(), 0);
  }
}
float AdamW::step() {
  double sum = 0;
  for (auto &[name, p] : parameters_) {
    ensure(p->grad.size() == p->data.size(), "Missing parameter gradients");
    for (float g : p->grad) {
      ensure(std::isfinite(g), "Nonfinite gradient");
      sum += double(g) * g;
    }
  }
  float norm = std::sqrt(sum),
        factor = std::min(1.f, config.clip / (norm + 1e-6f));
  ++steps;
  double correction1 = 1 - std::pow(double(config.beta1), steps),
         correction2 = 1 - std::pow(double(config.beta2), steps);
  for (auto &[name, p] : parameters_)
    for (std::size_t i = 0; i < p->data.size(); ++i) {
      float g = p->grad[i] * factor;
      auto &m = first[name][i];
      auto &v = second[name][i];
      m = config.beta1 * m + (1 - config.beta1) * g;
      v = config.beta2 * v + (1 - config.beta2) * g * g;
      p->data[i] *= 1 - config.learning_rate * config.weight_decay;
      p->data[i] -= config.learning_rate * (m / correction1) /
                    (std::sqrt(v / correction2) + config.epsilon);
      ensure(std::isfinite(p->data[i]), "Nonfinite parameter update");
    }
  return norm;
}
void save_checkpoint(const std::string &path, LQNet &m, const AdamW &o) {
  const auto temporary = path + ".tmp";
  std::ofstream f(temporary, std::ios::binary);
  if (!f)
    throw std::runtime_error("Cannot open checkpoint output");
  f.write("OWLQ0001", 8);
  const auto &c = m.config();
  for (int v : {c.width, c.dim, c.queries, c.classes, int(c.local_detail),
                int(c.full_resolution_detail), int(c.spatial_query)})
    write(f, std::int32_t(v));
  write(f, c.seed);
  write(f, o.steps);
  for (float v : {o.config.learning_rate, o.config.weight_decay, o.config.beta1,
                  o.config.beta2, o.config.epsilon, o.config.clip})
    write(f, v);
  write(f, std::uint32_t(m.parameters().size()));
  for (auto &[name, p] : m.parameters()) {
    write(f, std::uint32_t(name.size()));
    f.write(name.data(), name.size());
    write(f, std::uint64_t(p->data.size()));
    vector_write(f, p->data);
    vector_write(f, o.first.at(name));
    vector_write(f, o.second.at(name));
  }
  f.flush();
  if (!f)
    throw std::runtime_error("Checkpoint write failed");
  f.close();
  if (!f)
    throw std::runtime_error("Checkpoint close failed");
  // Same-directory rename preserves the prior checkpoint on a partial write.
  std::filesystem::rename(temporary, path);
}
void load_checkpoint(const std::string &path, LQNet &m, AdamW &o) {
  std::ifstream f(path, std::ios::binary);
  char magic[8];
  f.read(magic, 8);
  ensure(f && std::string(magic, 8) == "OWLQ0001", "Unsupported checkpoint");
  const auto &c = m.config();
  for (int expected :
       {c.width, c.dim, c.queries, c.classes, int(c.local_detail),
        int(c.full_resolution_detail), int(c.spatial_query)}) {
    std::int32_t actual;
    read(f, actual);
    ensure(expected == actual, "Checkpoint architecture mismatch");
  }
  std::uint32_t seed;
  read(f, seed);
  ensure(seed == c.seed, "Checkpoint seed mismatch");
  std::uint64_t steps;
  read(f, steps);
  for (float expected :
       {o.config.learning_rate, o.config.weight_decay, o.config.beta1,
        o.config.beta2, o.config.epsilon, o.config.clip}) {
    float actual;
    read(f, actual);
    ensure(actual == expected, "Checkpoint optimizer mismatch");
  }
  std::uint32_t count;
  read(f, count);
  ensure(count == m.parameters().size(), "Checkpoint parameter count mismatch");
  std::map<std::string, std::vector<float>> data, first, second;
  for (std::size_t k = 0; k < count; ++k) {
    std::uint32_t len;
    read(f, len);
    ensure(len > 0 && len <= 256, "Invalid checkpoint name");
    std::string name(len, '\0');
    f.read(name.data(), len);
    ensure(m.parameters().count(name) && !data.count(name),
           "Unknown/duplicate checkpoint parameter");
    std::uint64_t n;
    read(f, n);
    ensure(n == m.parameters().at(name)->data.size(),
           "Checkpoint tensor mismatch");
    data[name].resize(n);
    first[name].resize(n);
    second[name].resize(n);
    vector_read(f, data[name]);
    vector_read(f, first[name]);
    vector_read(f, second[name]);
    for (float v : second[name])
      ensure(v >= 0, "Invalid second moment");
  }
  ensure(f.peek() == std::char_traits<char>::eof(),
         "Trailing checkpoint content");
  for (auto &[name, p] : m.parameters())
    p->data = std::move(data[name]);
  o.first = std::move(first);
  o.second = std::move(second);
  o.steps = steps;
}
} // namespace ocuweave::nn
