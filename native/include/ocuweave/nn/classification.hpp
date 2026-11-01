#pragma once
#include "ocuweave/nn/lqnet.hpp"
#include <string>
namespace ocuweave::nn {
// A separately trained classification baseline using the LQNet residual
// encoder. Vessel masks are never used as disease labels or classifier inputs.
class Classifier {
public:
  Classifier(int width = 24, int classes = 4, std::uint32_t seed = 42);
  T forward(const T &images);
  void training(bool enabled) { model_.training(enabled); }
  LQNet &state() { return model_; }
  std::map<std::string, T> &parameters() { return model_.parameters(); }

private:
  LQNet model_;
};
T cross_entropy(const T &logits, const std::vector<int> &labels);
T image_subset(const T &images, const std::vector<std::size_t> &indices);
std::vector<std::size_t> minibatch_indices(std::size_t count,
                                           std::size_t batch_size,
                                           std::uint64_t step,
                                           std::uint32_t seed);
std::vector<int> read_labels(const std::string &path,
                             const std::vector<std::string> &ids, int classes);
} // namespace ocuweave::nn
