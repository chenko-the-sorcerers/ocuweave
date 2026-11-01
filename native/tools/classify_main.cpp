#include "ocuweave/nn/classification.hpp"
#include "ocuweave/nn/cuda.hpp"
#include "ocuweave/nn/io.hpp"
#include <algorithm>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <set>
#include <sstream>
using namespace ocuweave::nn;
namespace {
std::string quote(const std::string &s) {
  std::ostringstream o;
  o << '"';
  for (unsigned char c : s) {
    if (c < 32)
      throw std::invalid_argument("Control character in path/ID");
    if (c == '"' || c == '\\')
      o << '\\';
    o << c;
  }
  o << '"';
  return o.str();
}
void write(const std::filesystem::path &p, const std::string &s) {
  std::ofstream f(p);
  f << s << '\n';
  if (!f)
    throw std::runtime_error("Cannot write receipt");
}
std::string evaluate(Classifier &model, const Batch &b,
                     const std::vector<int> &labels, int batch_size,
                     const std::filesystem::path &output) {
  model.training(false);
  const int K = model.state().config().classes;
  std::vector<std::vector<int>> confusion(K, std::vector<int>(K));
  std::ofstream f(output);
  f << "sample_id,label";
  for (int j = 0; j < K; ++j)
    f << ",p" << j;
  f << '\n' << std::setprecision(9);
  int correct = 0;
  for (std::size_t first = 0; first < b.ids.size(); first += batch_size) {
    std::vector<std::size_t> index;
    for (std::size_t j = first; j < std::min(b.ids.size(), first + batch_size);
         ++j)
      index.push_back(j);
    auto p = softmax(model.forward(image_subset(b.image, index)), 1);
    for (std::size_t row = 0; row < index.size(); ++row) {
      auto i = index[row];
      auto begin = p->data.begin() + row * K;
      int predicted = std::max_element(begin, begin + K) - begin;
      ++confusion[labels[i]][predicted];
      correct += predicted == labels[i];
      f << b.ids[i] << ',' << labels[i];
      for (int j = 0; j < K; ++j)
        f << ',' << begin[j];
      f << '\n';
    }
  }
  f.close();
  if (!f)
    throw std::runtime_error("Prediction CSV write failed");
  std::ostringstream s;
  s << std::setprecision(10) << "{\"samples\":" << labels.size()
    << ",\"accuracy\":" << double(correct) / labels.size()
    << ",\"confusion_true_rows_predicted_columns\":[";
  for (int i = 0; i < K; ++i) {
    if (i)
      s << ',';
    s << '[';
    for (int j = 0; j < K; ++j) {
      if (j)
        s << ',';
      s << confusion[i][j];
    }
    s << ']';
  }
  s << "]}";
  return s.str();
}
} // namespace
int main(int argc, char **argv) {
  try {
    std::map<std::string, std::string> a;
    const std::set<std::string> keys = {
        "--train",   "--train-labels", "--validation", "--validation-labels",
        "--output",  "--steps",        "--batch-size", "--width",
        "--classes", "--seed",         "--device",     "--resume",
        "--lr"};
    for (int i = 1; i < argc; ++i) {
      std::string k = argv[i];
      if (k == "--help") {
        std::cout
            << "ocuweave-classify --train BATCH --train-labels TSV "
               "--validation BATCH --validation-labels TSV --output NEW_DIR "
               "[--steps 600 --batch-size 4 --device cpu|cuda --width 24 "
               "--classes 4 --seed 42 --lr 0.001 --resume CKPT]\n";
        return 0;
      }
      if (!keys.count(k) || a.count(k) || i + 1 >= argc)
        throw std::invalid_argument("Unknown/duplicate/missing argument");
      a[k] = argv[++i];
    }
    auto get = [&](const std::string &k, const std::string &d) {
      return a.count(k) ? a.at(k) : d;
    };
    auto integer = [&](const std::string &k, const std::string &d) {
      auto s = get(k, d);
      std::size_t n;
      int x = std::stoi(s, &n);
      if (n != s.size())
        throw std::invalid_argument("Invalid integer");
      return x;
    };
    for (const auto &k : {"--train", "--train-labels", "--validation",
                          "--validation-labels", "--output"})
      if (!a.count(k))
        throw std::invalid_argument(std::string("Missing ") + k);
    auto output = std::filesystem::path(a.at("--output"));
    if (std::filesystem::exists(output))
      throw std::invalid_argument("Preserve existing output");
    auto device = get("--device", "cpu");
    if (device != "cpu" && device != "cuda")
      throw std::invalid_argument("Unknown device");
    set_cuda(device == "cuda");
    int K = integer("--classes", "4"), width = integer("--width", "24"),
        seed = integer("--seed", "42"), steps = integer("--steps", "600"),
        batch_size = integer("--batch-size", "4");
    if (K < 2 || K > 64 || width < 1 || width > 256 || seed < 0 || steps < 0 ||
        steps > 1000000 || batch_size < 1)
      throw std::invalid_argument("Invalid training budget");
    auto train = load_batch(a.at("--train")),
         validation = load_batch(a.at("--validation"));
    if (train.partition != 1 || validation.partition != 2)
      throw std::invalid_argument(
          "Only train/validation partitions allowed; test is prohibited");
    if (batch_size > int(train.ids.size()))
      throw std::invalid_argument("Batch size exceeds training set");
    if (train.image->shape[2] != validation.image->shape[2] ||
        train.image->shape[3] != validation.image->shape[3])
      throw std::invalid_argument("Training/validation geometry differs");
    std::set<std::string> ids(train.ids.begin(), train.ids.end());
    for (auto &id : validation.ids)
      if (ids.count(id))
        throw std::invalid_argument("Cross-split sample ID");
    auto y = read_labels(a.at("--train-labels"), train.ids, K),
         yv = read_labels(a.at("--validation-labels"), validation.ids, K);
    std::vector<int> counts(K);
    for (auto label : y)
      ++counts[label];
    if (*std::min_element(counts.begin(), counts.end()) == 0)
      throw std::invalid_argument("Each disease class needs training examples");
    Classifier model(width, K, seed);
    AdamWConfig opt_cfg;
    std::size_t used;
    auto raw = get("--lr", "0.001");
    opt_cfg.learning_rate = std::stof(raw, &used);
    if (used != raw.size())
      throw std::invalid_argument("Invalid learning rate");
    AdamW opt(model.parameters(), opt_cfg);
    if (a.count("--resume"))
      load_checkpoint(a.at("--resume"), model.state(), opt);
    std::filesystem::create_directories(output);
    std::ostringstream protocol;
    protocol << "{\"version\":\"" << OCUWEAVE_VERSION
             << "\",\"source_commit\":\"" << OCUWEAVE_COMMIT
             << "\",\"source_dirty\":" << OCUWEAVE_DIRTY
             << ",\"model\":\"LQNet residual encoder with global-average "
                "disease head\",\"pretrained\":false,\"device\":"
             << quote(device) << ",\"seed\":" << seed << ",\"classes\":" << K
             << ",\"width\":" << width << ",\"batch_size\":" << batch_size
             << ",\"start_step\":" << opt.steps
             << ",\"additional_steps\":" << steps
             << ",\"augmentation\":\"none\",\"objective\":\"multiclass cross "
                "entropy\",\"test_evaluated\":false,\"arguments\":{";
    bool first = true;
    for (auto &[k, v] : a) {
      if (!first)
        protocol << ',';
      first = false;
      protocol << quote(k) << ':' << quote(v);
    }
    protocol << "}}";
    write(output / "protocol.json", protocol.str());
    save_checkpoint((output / "initial.ckpt").string(), model.state(), opt);
    auto start = std::chrono::steady_clock::now();
    std::ofstream history(output / "history.jsonl");
    model.training(true);
    for (int step = 0; step < steps; ++step) {
      auto index = minibatch_indices(y.size(), batch_size, opt.steps, seed);
      std::vector<int> labels;
      for (auto i : index)
        labels.push_back(y[i]);
      auto loss = cross_entropy(model.forward(image_subset(train.image, index)),
                                labels);
      backward(loss);
      auto norm = opt.step();
      history << "{\"step\":" << opt.steps << ",\"loss\":" << loss->data[0]
              << ",\"gradient_norm_before_clip\":" << norm << "}\n";
      history.flush();
      if (opt.steps % 25 == 0 || step == steps - 1) {
        save_checkpoint((output / "latest.ckpt").string(), model.state(), opt);
        std::cout << "step " << opt.steps << " loss " << loss->data[0]
                  << std::endl;
      }
    }
    save_checkpoint((output / "latest.ckpt").string(), model.state(), opt);
    auto train_metrics =
        evaluate(model, train, y, batch_size, output / "train-predictions.csv");
    auto val_metrics = evaluate(model, validation, yv, batch_size,
                                output / "validation-predictions.csv");
    Classifier reload(width, K, seed);
    AdamW reload_opt(reload.parameters(), opt_cfg);
    load_checkpoint((output / "latest.ckpt").string(), reload.state(),
                    reload_opt);
    reload.training(false);
    auto sample = image_subset(validation.image, {0});
    if (model.forward(sample)->data != reload.forward(sample)->data)
      throw std::runtime_error("Classifier checkpoint reload differs");
    double elapsed =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - start)
            .count();
    write(output / "metrics.json", "{\"train\":" + train_metrics +
                                       ",\"validation\":" + val_metrics +
                                       ",\"test_evaluated\":false,\"clinical_"
                                       "validity_established\":false}");
    write(output / "status.json",
          "{\"status\":\"completed_requires_review\",\"checkpoint_reload_"
          "matches\":true,\"elapsed_seconds\":" +
              std::to_string(elapsed) + "}");
    std::cout << "Validation " << val_metrics << '\n';
    return 0;
  } catch (const std::exception &e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
