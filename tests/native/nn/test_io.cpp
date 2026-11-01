#include "ocuweave/nn/io.hpp"
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
    std::string path = argc > 1 ? argv[1] : "batch-test.bin";
    auto b = synthetic_batch(16, 1, 1);
    save_batch(path, b);
    auto copy = load_batch(path);
    check(copy.image->data == b.image->data &&
              copy.target->data == b.target->data && copy.ids == b.ids,
          "Batch roundtrip");
    auto report = metric_report(b.target, b, .5, true);
    check(report.find("\"dice\":1") != std::string::npos &&
              report.find("\"cldice\":1") != std::string::npos,
          "Perfect overlap/skeleton oracle");
    auto empty = zeros(b.target->shape);
    report = metric_report(empty, b, .5, true);
    check(report.find("\"dice\":0") != std::string::npos &&
              report.find("\"cldice\":0") != std::string::npos,
          "Empty prediction oracle");
    b.target = empty;
    report = metric_report(empty, b, .5, true);
    check(report.find("\"dice\":1") != std::string::npos &&
              report.find("\"cldice\":1") != std::string::npos,
          "Empty pair convention");
    b.available[0] = 0;
    report = metric_report(empty, b);
    check(report.find("\"dice\":null") != std::string::npos,
          "Missing annotations excluded");
    std::remove(path.c_str());
    std::cout << "Native IO and metric oracles passed\n";
    return 0;
  } catch (const std::exception &e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
