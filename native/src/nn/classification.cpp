#include "ocuweave/nn/classification.hpp"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <numeric>
#include <stdexcept>
namespace ocuweave::nn {
namespace {
void require(bool v, const char *m) {
  if (!v)
    throw std::invalid_argument(m);
}
std::uint32_t random(std::uint32_t &s) {
  s ^= s << 13;
  s ^= s >> 17;
  s ^= s << 5;
  return s;
}
ModelConfig config(int width, int classes, std::uint32_t seed) {
  ModelConfig c;
  c.width = width;
  c.dim = 8;
  c.queries = 2;
  c.classes = classes;
  c.local_detail = false;
  c.seed = seed;
  return c;
}
} // namespace
Classifier::Classifier(int width, int classes, std::uint32_t seed)
    : model_(config(width, classes, seed)) {
  require(classes >= 2, "At least two disease classes required");
  auto &p = model_.parameters();
  for (auto it = p.begin(); it != p.end();) {
    if (it->first.rfind("encoder.", 0) != 0)
      it = p.erase(it);
    else
      ++it;
  }
  auto w = zeros({1, std::size_t(4 * width), std::size_t(classes)}, true);
  std::uint32_t state = seed ^ 0x9e3779b9u;
  if (!state)
    state = 1;
  for (auto &v : w->data)
    v = (2 * ((double(random(state)) + .5) / 4294967296.) - 1) /
        std::sqrt(float(4 * width));
  p["classifier.weight"] = w;
  p["classifier.bias"] = zeros({1, 1, std::size_t(classes)}, true);
}
T Classifier::forward(const T &images) {
  auto z = model_.encode(images).back();
  auto pooled = scale(reduce_sum(reduce_sum(z, 3), 2),
                      1.f / float(z->shape[2] * z->shape[3]));
  auto &p = parameters();
  auto logits = add(matmul(reshape(pooled, {z->shape[0], 1, z->shape[1]}),
                           p.at("classifier.weight")),
                    p.at("classifier.bias"));
  return reshape(logits, {z->shape[0], std::size_t(model_.config().classes)});
}
T cross_entropy(const T &logits, const std::vector<int> &labels) {
  require(logits->shape.size() == 2 && labels.size() == logits->shape[0],
          "Classification label/shape mismatch");
  auto n = logits->shape[0], c = logits->shape[1];
  std::vector<float> gradient(logits->data.size());
  double loss = 0;
  for (std::size_t b = 0; b < n; ++b) {
    require(labels[b] >= 0 && std::size_t(labels[b]) < c,
            "Unknown disease class");
    const auto begin = logits->data.begin() + b * c;
    float maximum = *std::max_element(begin, begin + c);
    require(std::isfinite(maximum), "Nonfinite classifier logits");
    double sum = 0;
    for (std::size_t j = 0; j < c; ++j) {
      require(std::isfinite(begin[j]), "Nonfinite classifier logits");
      sum += std::exp(double(begin[j]) - maximum);
    }
    loss += std::log(sum) + maximum - begin[labels[b]];
    for (std::size_t j = 0; j < c; ++j)
      gradient[b * c + j] =
          (std::exp(double(begin[j]) - maximum) / sum - (int(j) == labels[b])) /
          n;
  }
  auto out = tensor({1}, {float(loss / n)}, logits->requires_grad);
  if (out->requires_grad) {
    out->parents = {logits};
    out->vjp = [logits, gradient](const auto &g) {
      for (std::size_t i = 0; i < gradient.size(); ++i)
        logits->grad[i] += g[0] * gradient[i];
    };
  }
  return out;
}
T image_subset(const T &images, const std::vector<std::size_t> &indices) {
  require(images->shape.size() == 4 && !indices.empty(),
          "Image minibatch requires NCHW inputs and nonempty indices");
  auto shape = images->shape;
  auto stride = images->data.size() / shape[0];
  shape[0] = indices.size();
  auto result = zeros(shape);
  for (std::size_t b = 0; b < indices.size(); ++b) {
    require(indices[b] < images->shape[0], "Image index out of range");
    std::copy_n(images->data.begin() + indices[b] * stride, stride,
                result->data.begin() + b * stride);
  }
  return result;
}
std::vector<std::size_t> minibatch_indices(std::size_t count,
                                           std::size_t batch_size,
                                           std::uint64_t step,
                                           std::uint32_t seed) {
  require(count > 0 && batch_size > 0 && batch_size <= count,
          "Invalid minibatch extent");
  auto batches = (count + batch_size - 1) / batch_size;
  auto epoch = step / batches, index = step % batches;
  std::vector<std::size_t> order(count);
  std::iota(order.begin(), order.end(), 0);
  std::uint32_t state =
      seed ^ std::uint32_t(epoch) ^ std::uint32_t(epoch >> 32) ^ 0x85ebca6bu;
  if (!state)
    state = 1;
  for (std::size_t i = count - 1; i > 0; --i)
    std::swap(order[i], order[random(state) % (i + 1)]);
  std::size_t first = std::size_t(index) * batch_size;
  auto last = std::min(count, first + batch_size);
  return {order.begin() + first, order.begin() + last};
}
std::vector<int> read_labels(const std::string &path,
                             const std::vector<std::string> &ids, int classes) {
  std::ifstream f(path);
  require(bool(f), "Cannot open classification labels");
  std::string line;
  std::getline(f, line);
  require(line == "sample_id\tlabel", "Unknown label format");
  std::vector<int> result;
  for (const auto &id : ids) {
    require(bool(std::getline(f, line)), "Missing disease label");
    auto tab = line.find('\t');
    require(tab != std::string::npos && line.substr(0, tab) == id,
            "Image/label identity mismatch");
    std::size_t used;
    auto raw = line.substr(tab + 1);
    int label = std::stoi(raw, &used);
    require(used == raw.size() && label >= 0 && label < classes,
            "Invalid disease label");
    result.push_back(label);
  }
  require(!std::getline(f, line), "Extra disease labels");
  return result;
}
} // namespace ocuweave::nn
