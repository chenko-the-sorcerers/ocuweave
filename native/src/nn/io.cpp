#include "ocuweave/nn/io.hpp"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>
namespace ocuweave::nn {
namespace {
void require(bool b, const char *m) {
  if (!b)
    throw std::invalid_argument(m);
}
void endian() {
  std::uint32_t n = 1;
  require(*reinterpret_cast<unsigned char *>(&n) == 1 &&
              std::numeric_limits<float>::is_iec559,
          "Little-endian IEEE float32 platform required");
}
template <class V> void read(std::istream &f, V &v) {
  f.read(reinterpret_cast<char *>(&v), sizeof(v));
  if (!f)
    throw std::runtime_error("Truncated native batch");
}
template <class V> void write(std::ostream &f, const V &v) {
  f.write(reinterpret_cast<const char *>(&v), sizeof(v));
}
std::string quote(const std::string &s) {
  std::ostringstream o;
  o << '"';
  for (unsigned char c : s) {
    if (c == '"' || c == '\\')
      o << '\\' << c;
    else if (c < 32)
      o << "\\u00" << std::hex << std::setw(2) << std::setfill('0') << int(c)
        << std::dec;
    else
      o << c;
  }
  o << '"';
  return o.str();
}
std::vector<unsigned char> skeleton(std::vector<unsigned char> a, int H,
                                    int W) {
  bool changed = true;
  while (changed) {
    changed = false;
    for (int phase = 0; phase < 2; ++phase) {
      std::vector<int> remove;
      for (int y = 1; y < H - 1; ++y)
        for (int x = 1; x < W - 1; ++x) {
          int k = y * W + x;
          if (!a[k])
            continue;
          int p[8] = {a[k - W], a[k - W + 1], a[k + 1], a[k + W + 1],
                      a[k + W], a[k + W - 1], a[k - 1], a[k - W - 1]},
              neighbors = 0, transitions = 0;
          for (int i = 0; i < 8; ++i) {
            neighbors += p[i];
            transitions += p[i] == 0 && p[(i + 1) % 8] == 1;
          }
          if (neighbors < 2 || neighbors > 6 || transitions != 1)
            continue;
          bool condition =
              phase == 0 ? (p[0] * p[2] * p[4] == 0 && p[2] * p[4] * p[6] == 0)
                         : (p[0] * p[2] * p[6] == 0 && p[0] * p[4] * p[6] == 0);
          if (condition)
            remove.push_back(k);
        }
      for (int k : remove)
        a[k] = 0;
      changed = changed || !remove.empty();
    }
  }
  return a;
}
} // namespace
Batch load_batch(const std::string &path) {
  endian();
  std::ifstream f(path, std::ios::binary);
  char magic[8];
  f.read(magic, 8);
  require(f && std::string(magic, 8) == "OWBATCH1", "Unsupported native batch");
  std::uint32_t n, c, h, w, partition;
  read(f, n);
  read(f, c);
  read(f, h);
  read(f, w);
  read(f, partition);
  require(n > 0 && n <= 512 && c > 0 && c <= 32 && h >= 16 && w >= 16 &&
              h <= 4096 && w <= 4096 &&
              std::uint64_t(n) * (3 + c + 1) * h * w <= 150000000,
          "Native batch extents exceed limit");
  require(partition >= 1 && partition <= 3,
          "Partition must be train/validation/test");
  Batch b;
  b.partition = partition;
  for (std::uint32_t i = 0; i < n; ++i) {
    std::uint32_t len;
    read(f, len);
    require(len > 0 && len <= 256, "Invalid sample ID length");
    std::string id(len, '\0');
    f.read(id.data(), len);
    require(std::find(b.ids.begin(), b.ids.end(), id) == b.ids.end(),
            "Duplicate sample ID");
    b.ids.push_back(id);
  }
  b.image = zeros({n, 3, h, w});
  b.target = zeros({n, c, h, w});
  b.valid = zeros({n, 1, h, w});
  b.available.resize(n * c);
  for (auto t : {b.image, b.target, b.valid}) {
    f.read(reinterpret_cast<char *>(t->data.data()), t->data.size() * 4);
    require(bool(f), "Truncated tensor payload");
    for (auto x : t->data)
      require(std::isfinite(x) && x >= 0 && x <= 1,
              "Native batch values must be finite [0,1]");
  }
  f.read(reinterpret_cast<char *>(b.available.data()), b.available.size());
  require(bool(f), "Truncated availability");
  for (auto v : b.available)
    require(v <= 1, "Availability must be binary");
  for (auto v : b.valid->data)
    require(v == 0 || v == 1, "Validity must be binary");
  require(f.peek() == std::char_traits<char>::eof(), "Trailing batch bytes");
  return b;
}
void save_batch(const std::string &path, const Batch &b) {
  endian();
  std::ofstream f(path, std::ios::binary);
  if (!f)
    throw std::runtime_error("Cannot write batch");
  f.write("OWBATCH1", 8);
  for (auto v :
       {std::uint32_t(b.image->shape[0]), std::uint32_t(b.target->shape[1]),
        std::uint32_t(b.image->shape[2]), std::uint32_t(b.image->shape[3]),
        b.partition})
    write(f, v);
  for (auto &id : b.ids) {
    write(f, std::uint32_t(id.size()));
    f.write(id.data(), id.size());
  }
  for (auto t : {b.image, b.target, b.valid})
    f.write(reinterpret_cast<const char *>(t->data.data()), t->data.size() * 4);
  f.write(reinterpret_cast<const char *>(b.available.data()),
          b.available.size());
  if (!f)
    throw std::runtime_error("Batch write failed");
}
Batch synthetic_batch(int size, int classes, int count) {
  require(size >= 16 && size <= 256 && classes > 0 && classes <= 4 &&
              count > 0 && count <= 8,
          "Invalid synthetic demo geometry");
  Batch b;
  b.partition = 1;
  b.image =
      zeros({std::size_t(count), 3, std::size_t(size), std::size_t(size)});
  b.target = zeros({std::size_t(count), std::size_t(classes), std::size_t(size),
                    std::size_t(size)});
  b.valid =
      zeros({std::size_t(count), 1, std::size_t(size), std::size_t(size)});
  b.available.assign(count * classes, 1);
  std::fill(b.valid->data.begin(), b.valid->data.end(), 1);
  for (int n = 0; n < count; ++n) {
    b.ids.push_back("synthetic-" + std::to_string(n));
    for (int y = 0; y < size; ++y)
      for (int x = 0; x < size; ++x) {
        for (int k = 0; k < 3; ++k)
          b.image->data[((n * 3 + k) * size + y) * size + x] =
              .08 + .01 * ((x + y + k) % 3);
        for (int c = 0; c < classes; ++c) {
          float dx = x - (size * .3 + size * .1 * n),
                dy = y - (size * .3 + size * .12 * c);
          bool lesion = dx * dx + dy * dy < size * size * .025;
          if (lesion) {
            b.target->data[((n * classes + c) * size + y) * size + x] = 1;
            b.image->data[((n * 3 + c % 3) * size + y) * size + x] = .9;
          }
        }
      }
  }
  return b;
}
void write_predictions(const std::string &path, const T &p) {
  endian();
  std::ofstream f(path, std::ios::binary);
  require(p->shape.size() == 4, "Predictions must be NCHW");
  f.write("OWPROB01", 8);
  for (auto d : p->shape)
    write(f, std::uint32_t(d));
  f.write(reinterpret_cast<const char *>(p->data.data()), p->data.size() * 4);
  if (!f)
    throw std::runtime_error("Prediction write failed");
}
std::string metric_report(const T &p, const Batch &b, float threshold,
                          bool vessels) {
  require(p->shape == b.target->shape && std::isfinite(threshold) &&
              threshold >= 0 && threshold <= 1,
          "Invalid metric contract");
  std::ostringstream out;
  out << std::setprecision(10)
      << "{\"grid\":\"input\",\"area_denominator\":\"valid image pixels, not a "
         "verified retinal "
         "field\",\"physical_area_mm2\":null,\"early_detection_validated\":"
         "false,\"threshold\":"
      << threshold << ",\"rows\":[";
  auto N = p->shape[0], C = p->shape[1], H = p->shape[2], W = p->shape[3],
       S = H * W;
  bool first = true;
  for (std::size_t n = 0; n < N; ++n)
    for (std::size_t c = 0; c < C; ++c) {
      if (!first)
        out << ',';
      first = false;
      out << "{\"sample_id\":" << quote(b.ids[n]) << ",\"class_index\":" << c
          << ",\"annotation_available\":"
          << (b.available[n * C + c] ? "true" : "false");
      std::size_t tp = 0, fp = 0, fn = 0, area = 0, denominator = 0, sumx = 0,
                  sumy = 0;
      int xmin = W, ymin = H, xmax = -1, ymax = -1;
      std::vector<unsigned char> pred((H + 2) * (W + 2)), truth(pred.size());
      for (std::size_t y = 0; y < H; ++y)
        for (std::size_t x = 0; x < W; ++x) {
          auto i = y * W + x, k = (n * C + c) * S + i;
          if (!b.valid->data[n * S + i])
            continue;
          require(std::isfinite(p->data[k]) && p->data[k] >= 0 &&
                      p->data[k] <= 1,
                  "Invalid probability");
          bool a = p->data[k] >= threshold, t = b.target->data[k] > .5;
          ++denominator;
          if (a) {
            ++area;
            sumx += x;
            sumy += y;
            xmin = std::min(xmin, int(x));
            xmax = std::max(xmax, int(x));
            ymin = std::min(ymin, int(y));
            ymax = std::max(ymax, int(y));
          }
          tp += a && t;
          fp += a && !t;
          fn += !a && t;
          pred[(y + 1) * (W + 2) + x + 1] = a;
          truth[(y + 1) * (W + 2) + x + 1] = t;
        }
      require(denominator > 0, "Empty valid image area");
      out << ",\"predicted_area_pixels\":" << area
          << ",\"valid_image_pixels\":" << denominator
          << ",\"predicted_area_percent\":" << 100. * area / denominator
          << ",\"centroid_xy\":";
      if (area)
        out << '[' << double(sumx) / area << ',' << double(sumy) / area << ']';
      else
        out << "null";
      out << ",\"bbox_xyxy_inclusive\":";
      if (area)
        out << '[' << xmin << ',' << ymin << ',' << xmax << ',' << ymax << ']';
      else
        out << "null";
      if (b.available[n * C + c]) {
        out << ",\"tp\":" << tp << ",\"fp\":" << fp << ",\"fn\":" << fn
            << ",\"dice\":"
            << (2 * tp + fp + fn ? 2. * tp / (2 * tp + fp + fn) : 1)
            << ",\"iou\":" << (tp + fp + fn ? double(tp) / (tp + fp + fn) : 1)
            << ",\"false_positive_area_percent\":" << 100. * fp / denominator;
        if (vessels) {
          auto ps = skeleton(pred, H + 2, W + 2),
               ts = skeleton(truth, H + 2, W + 2);
          std::size_t pc = 0, tc = 0, ph = 0, th = 0;
          for (std::size_t k = 0; k < ps.size(); ++k) {
            pc += ps[k];
            tc += ts[k];
            ph += ps[k] && truth[k];
            th += ts[k] && pred[k];
          }
          double precision = pc ? double(ph) / pc : 0,
                 recall = tc ? double(th) / tc : 0,
                 cl = pc == 0 && tc == 0
                          ? 1
                          : (precision + recall
                                 ? 2 * precision * recall / (precision + recall)
                                 : 0);
          out << ",\"cldice\":" << cl << ",\"skeleton_precision\":" << precision
              << ",\"skeleton_recall\":" << recall;
        }
      } else
        out << ",\"dice\":null,\"iou\":null";
      out << '}';
    }
  out << "]}";
  return out.str();
}
} // namespace ocuweave::nn
