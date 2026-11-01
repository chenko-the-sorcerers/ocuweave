#pragma once
#include "ocuweave/nn/tensor.hpp"
#include <cstdint>
#include <map>
#include <string>
namespace ocuweave::nn {
struct ModelConfig {
  int width = 24, dim = 48, queries = 8, classes = 1;
  // Independent ablations; baseline matches the published image-only LQNet
  // graph.
  bool local_detail = true, full_resolution_detail = false,
       spatial_query = false;
  std::uint32_t seed = 42;
};
class LQNet {
public:
  explicit LQNet(ModelConfig config);
  T forward(const T &image);
  std::vector<T> encode(const T &image);
  const ModelConfig &config() const { return config_; }
  std::map<std::string, T> &parameters() { return parameters_; }
  void training(bool enabled);
  std::size_t parameter_count() const;

private:
  ModelConfig config_;
  std::map<std::string, T> parameters_;
  T parameter(const std::string &, Shape, float bound, bool zero = false,
              bool normal = false);
  void convolution(const std::string &, int, int, int, int groups = 1,
                   bool zero = false);
  void norm_parameters(const std::string &, int);
  void linear_parameters(const std::string &, int, int);
  void mixer_parameters(const std::string &, int);
  T convolution_forward(const std::string &, const T &, int stride = 1,
                        int padding = 0, int groups = 1);
  T norm_forward(const std::string &, const T &);
  T linear_forward(const std::string &, const T &);
  T mixer_forward(const std::string &, const T &);
  T head_forward(const std::string &, const std::vector<T> &, std::size_t,
                 std::size_t);
};
struct AdamWConfig {
  float learning_rate = .001f, weight_decay = .0001f, beta1 = .9f,
        beta2 = .999f, epsilon = 1e-8f, clip = 1.f;
};
class AdamW {
public:
  AdamW(std::map<std::string, T> &parameters, AdamWConfig config = {});
  float step();
  std::uint64_t steps = 0;
  AdamWConfig config;
  std::map<std::string, std::vector<float>> first, second;

private:
  std::map<std::string, T> &parameters_;
};
void save_checkpoint(const std::string &path, LQNet &model,
                     const AdamW &optimizer);
void load_checkpoint(const std::string &path, LQNet &model, AdamW &optimizer);
} // namespace ocuweave::nn
