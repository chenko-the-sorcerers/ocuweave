#pragma once
#include "ocuweave/nn/tensor.hpp"
#include <cstdint>
#include <string>
namespace ocuweave::nn {
struct Batch {
  T image, target, valid;
  std::vector<unsigned char> available;
  std::vector<std::string> ids;
  std::uint32_t partition;
};
// Little-endian IEEE float32; exact preprocessed tensors, no image decoder
// dependency.
Batch load_batch(const std::string &path);
void save_batch(const std::string &path, const Batch &batch);
Batch synthetic_batch(int size = 32, int classes = 1, int count = 2);
void write_predictions(const std::string &path, const T &probabilities);
std::string metric_report(const T &probabilities, const Batch &batch,
                          float threshold = .5f, bool vessels = false);
} // namespace ocuweave::nn
