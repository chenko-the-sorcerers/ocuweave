#include "ocuweave/nn/cuda.hpp"
#include "ocuweave/nn/io.hpp"
#include "ocuweave/nn/lqnet.hpp"
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
using namespace ocuweave::nn;
// Inference reads architecture/optimizer metadata from a native checkpoint,
// never executable pickle.
int main(int argc, char **argv) {
  try {
    if (argc != 5) {
      std::cerr
          << "Usage: ocuweave-predict CHECKPOINT BATCH OUTPUT_PROB cpu|cuda\n";
      return 2;
    }
    if (std::filesystem::exists(argv[3]))
      throw std::invalid_argument("Output exists");
    std::string device = argv[4];
    if (device != "cpu" && device != "cuda")
      throw std::invalid_argument("Invalid device");
    set_cuda(device == "cuda");
    std::ifstream f(argv[1], std::ios::binary);
    char magic[8];
    f.read(magic, 8);
    if (!f || std::string(magic, 8) != "OWLQ0001")
      throw std::invalid_argument("Invalid checkpoint");
    auto read = [&](auto &x) {
      f.read(reinterpret_cast<char *>(&x), sizeof(x));
      if (!f)
        throw std::runtime_error("Truncated checkpoint");
    };
    ModelConfig c;
    std::int32_t values[7];
    for (auto &v : values)
      read(v);
    c.width = values[0];
    c.dim = values[1];
    c.queries = values[2];
    c.classes = values[3];
    for (int i = 4; i < 7; ++i)
      if (values[i] != 0 && values[i] != 1)
        throw std::invalid_argument("Invalid architecture flag");
    c.local_detail = values[4];
    c.full_resolution_detail = values[5];
    c.spatial_query = values[6];
    read(c.seed);
    std::uint64_t steps;
    read(steps);
    AdamWConfig ac;
    for (float *p : {&ac.learning_rate, &ac.weight_decay, &ac.beta1, &ac.beta2,
                     &ac.epsilon, &ac.clip})
      read(*p);
    f.close();
    LQNet model(c);
    AdamW opt(model.parameters(), ac);
    load_checkpoint(argv[1], model, opt);
    model.training(false);
    auto batch = load_batch(argv[2]);
    if (batch.target->shape[1] != std::size_t(c.classes))
      throw std::invalid_argument("Class mismatch");
    write_predictions(argv[3], sigmoid(model.forward(batch.image)));
    std::cout << "Native inference completed\n";
    return 0;
  } catch (const std::exception &e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
