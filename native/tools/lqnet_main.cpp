#include "ocuweave/nn/cuda.hpp"
#include "ocuweave/nn/io.hpp"
#include "ocuweave/nn/lqnet.hpp"
#include <algorithm>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
using namespace ocuweave::nn;
namespace {
void file(const std::filesystem::path &p, const std::string &s) {
  std::ofstream f(p);
  f << s << '\n';
  if (!f)
    throw std::runtime_error("Failed writing " + p.string());
}
std::string text(const std::string &s) {
  std::ostringstream o;
  o << '"';
  for (unsigned char c : s) {
    if (c == '"' || c == '\\')
      o << '\\';
    if (c < 32)
      throw std::invalid_argument("Control character in argument");
    o << c;
  }
  o << '"';
  return o.str();
}
} // namespace
int main(int argc, char **argv) {
  try {
    std::map<std::string, std::string> args;
    for (int i = 1; i < argc; ++i) {
      std::string k = argv[i];
      if (k == "--help") {
        std::cout
            << "ocuweave-lqnet --output DIR [--batch FILE | --synthetic true] "
               "[--steps 200] [--device cpu|cuda] [--classes 1] [--width 24] "
               "[--dim 48] [--queries 8] [--seed 42] [--size 32] [--weights "
               "1,1,1,1] [--lr 0.001] [--variant "
               "baseline|detail_full|spatial_query] [--resume FILE] [--vessels "
               "true] [--save-every 25]\nExplicit analytic backward, custom "
               "AdamW; no pretrained weights or ML framework.\n";
        return 0;
      }
      if (k.rfind("--", 0) != 0 || i + 1 >= argc || args.count(k))
        throw std::invalid_argument("Expected unique --key value pairs");
      args[k] = argv[++i];
    }
    const std::vector<std::string> known = {
        "--output",  "--batch",     "--synthetic", "--steps",   "--device",
        "--classes", "--width",     "--dim",       "--queries", "--seed",
        "--size",    "--weights",   "--lr",        "--variant", "--resume",
        "--vessels", "--save-every"};
    for (auto &[k, v] : args)
      if (std::find(known.begin(), known.end(), k) == known.end())
        throw std::invalid_argument("Unknown option " + k);
    auto get = [&](const std::string &k, const std::string &d) {
      return args.count(k) ? args.at(k) : d;
    };
    auto integer = [&](const std::string &k, const std::string &d) {
      auto s = get(k, d);
      std::size_t used;
      int v = std::stoi(s, &used);
      if (used != s.size())
        throw std::invalid_argument("Invalid integer");
      return v;
    };
    auto output = get("--output", "");
    if (output.empty())
      throw std::invalid_argument("--output required");
    if (std::filesystem::exists(output))
      throw std::invalid_argument("Output exists; preserve previous run");
    auto device = get("--device", "cpu");
    if (device != "cpu" && device != "cuda")
      throw std::invalid_argument("Device must be cpu/cuda");
    set_cuda(device == "cuda");
    bool synthetic = get("--synthetic", "false") == "true";
    if (synthetic == bool(args.count("--batch")))
      throw std::invalid_argument(
          "Choose exactly one of --batch or --synthetic true");
    ModelConfig cfg;
    cfg.width = integer("--width", "24");
    cfg.dim = integer("--dim", "48");
    cfg.queries = integer("--queries", "8");
    cfg.classes = integer("--classes", "1");
    int seed = integer("--seed", "42");
    if (seed < 0)
      throw std::invalid_argument("Nonnegative seed required");
    cfg.seed = seed;
    auto variant = get("--variant", "baseline");
    if (variant != "baseline" && variant != "detail_full" &&
        variant != "spatial_query")
      throw std::invalid_argument("Unknown variant");
    cfg.full_resolution_detail = variant == "detail_full";
    cfg.spatial_query = variant == "spatial_query";
    auto batch = synthetic
                     ? synthetic_batch(integer("--size", "32"), cfg.classes)
                     : load_batch(args.at("--batch"));
    if (batch.partition != 1)
      throw std::invalid_argument(
          "Diagnostic fitting accepts training-only batches");
    if (batch.target->shape[1] != std::size_t(cfg.classes))
      throw std::invalid_argument("Class count differs from batch");
    int steps = integer("--steps", "200"),
        save_every = integer("--save-every", "25");
    if (steps < 0 || steps > 1000000 || save_every < 1)
      throw std::invalid_argument("Invalid step budget");
    std::vector<float> weights;
    if (args.count("--weights")) {
      std::istringstream s(args.at("--weights"));
      std::string v;
      while (std::getline(s, v, ',')) {
        std::size_t used;
        auto w = std::stof(v, &used);
        if (used != v.size())
          throw std::invalid_argument("Invalid positive weight");
        weights.push_back(w);
      }
      if (weights.size() != std::size_t(cfg.classes))
        throw std::invalid_argument("Weight/class count mismatch");
    }
    LQNet model(cfg);
    AdamWConfig ac;
    {
      auto s = get("--lr", "0.001");
      std::size_t used;
      ac.learning_rate = std::stof(s, &used);
      if (used != s.size())
        throw std::invalid_argument("Invalid learning rate");
    }
    AdamW optimizer(model.parameters(), ac);
    if (args.count("--resume"))
      load_checkpoint(args.at("--resume"), model, optimizer);
    std::filesystem::create_directories(output);
    std::filesystem::path dir(output);
    std::ostringstream protocol;
    protocol << "{\"software\":\"ocuweave-native\",\"version\":\""
             << OCUWEAVE_VERSION << "\",\"source_commit\":\"" << OCUWEAVE_COMMIT
             << "\",\"source_dirty\":" << OCUWEAVE_DIRTY
             << ",\"device\":" << text(device)
             << ",\"synthetic\":" << (synthetic ? "true" : "false")
             << ",\"purpose\":\"training-only diagnostic, not "
                "generalization\",\"parameters\":"
             << model.parameter_count() << ",\"start_step\":" << optimizer.steps
             << ",\"additional_steps\":" << steps
             << ",\"area_grid\":\"input\",\"arguments\":{";
    bool first = true;
    for (auto &[k, v] : args) {
      if (!first)
        protocol << ',';
      first = false;
      protocol << text(k) << ':' << text(v);
    }
    protocol << "}}";
    file(dir / "protocol.json", protocol.str());
    save_checkpoint((dir / "initial.ckpt").string(), model, optimizer);
    auto start = std::chrono::steady_clock::now();
    std::ofstream history(dir / "history.jsonl");
    for (int step = 1; step <= steps; ++step) {
      auto loss = observed_loss(model.forward(batch.image), batch.target,
                                batch.valid, batch.available, weights);
      backward(loss.total);
      float norm = optimizer.step();
      std::ostringstream row;
      row << "{\"step\":" << optimizer.steps
          << ",\"loss\":" << loss.total->data[0] << ",\"bce\":" << loss.bce
          << ",\"soft_dice\":" << loss.dice
          << ",\"gradient_norm_before_clip\":" << norm << '}';
      history << row.str() << '\n';
      history.flush();
      if (step % save_every == 0 || step == steps) {
        save_checkpoint((dir / "latest.ckpt").string(), model, optimizer);
        std::cout << row.str() << std::endl;
      }
    }
    save_checkpoint((dir / "latest.ckpt").string(), model, optimizer);
    model.training(false);
    auto probabilities = sigmoid(model.forward(batch.image));
    write_predictions((dir / "predictions.bin").string(), probabilities);
    file(dir / "metrics.json",
         metric_report(probabilities, batch, .5f,
                       get("--vessels", "false") == "true"));
    LQNet reloaded(cfg);
    AdamW reload_opt(reloaded.parameters(), ac);
    load_checkpoint((dir / "latest.ckpt").string(), reloaded, reload_opt);
    reloaded.training(false);
    auto again = sigmoid(reloaded.forward(batch.image));
    if (again->data != probabilities->data)
      throw std::runtime_error("Checkpoint reload mismatch");
    double seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - start)
            .count();
    file(dir / "status.json",
         "{\"status\":\"completed_requires_review\",\"checkpoint_reload_"
         "matches\":true,\"elapsed_seconds\":" +
             std::to_string(seconds) +
             ",\"validation_evaluated\":false,\"test_evaluated\":false}");
    std::cout << "Saved " << output << " in " << seconds << " seconds\n";
    return 0;
  } catch (const std::exception &e) {
    std::cerr << "ocuweave-lqnet: " << e.what() << '\n';
    return 1;
  }
}
