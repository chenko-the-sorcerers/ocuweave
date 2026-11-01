#include "ocuweave/nn/tensor.hpp"
#include "ocuweave/nn/cuda.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <unordered_set>

namespace ocuweave::nn {
namespace {
void require(bool value, const char *message) {
  if (!value)
    throw std::invalid_argument(message);
}
Shape strides(const Shape &s) {
  Shape r(s.size(), 1);
  for (std::size_t i = s.size(); i > 1; --i)
    r[i - 2] = r[i - 1] * s[i - 1];
  return r;
}
void accumulate(const T &t, std::size_t i, float g) {
  if (t->requires_grad)
    t->grad[i] += g;
}
T operation(Shape s, std::vector<T> parents) {
  bool train = false;
  for (const auto &p : parents)
    train = train || p->requires_grad;
  auto t = zeros(std::move(s), train);
  if (train)
    t->parents = std::move(parents);
  return t;
}
std::size_t broadcast_index(std::size_t index, const Shape &out,
                            const Shape &in) {
  std::size_t result = 0, step = 1;
  const auto offset = out.size() - in.size();
  for (std::size_t j = out.size(); j > 0; --j) {
    const auto coordinate = index % out[j - 1];
    index /= out[j - 1];
    if (j - 1 >= offset) {
      auto k = j - 1 - offset;
      result += (in[k] == 1 ? 0 : coordinate) * step;
      step *= in[k];
    }
  }
  return result;
}
T binary(const T &a, const T &b, bool product) {
  Shape shape(std::max(a->shape.size(), b->shape.size()), 1);
  for (std::size_t i = 0; i < shape.size(); ++i) {
    auto ai = i < a->shape.size() ? a->shape[a->shape.size() - 1 - i] : 1;
    auto bi = i < b->shape.size() ? b->shape[b->shape.size() - 1 - i] : 1;
    require(ai == bi || ai == 1 || bi == 1, "Incompatible broadcast shapes");
    shape[shape.size() - 1 - i] = std::max(ai, bi);
  }
  auto o = operation(shape, {a, b});
  for (std::size_t i = 0; i < o->data.size(); ++i) {
    auto x = broadcast_index(i, shape, a->shape),
         y = broadcast_index(i, shape, b->shape);
    o->data[i] = product ? a->data[x] * b->data[y] : a->data[x] + b->data[y];
  }
  if (o->requires_grad)
    o->vjp = [a, b, shape, product](const auto &g) {
      for (std::size_t i = 0; i < g.size(); ++i) {
        auto x = broadcast_index(i, shape, a->shape),
             y = broadcast_index(i, shape, b->shape);
        accumulate(a, x, g[i] * (product ? b->data[y] : 1));
        accumulate(b, y, g[i] * (product ? a->data[x] : 1));
      }
    };
  return o;
}
float logistic(float v) {
  return v >= 0 ? 1.f / (1.f + std::exp(-v))
                : std::exp(v) / (1.f + std::exp(v));
}
} // namespace
std::size_t elements(const Shape &s) {
  require(!s.empty(), "Tensor rank must be positive");
  std::size_t n = 1;
  for (auto d : s) {
    require(d > 0 && n <= std::numeric_limits<std::size_t>::max() / d,
            "Invalid tensor extent");
    n *= d;
  }
  return n;
}
Tensor::Tensor(Shape s, std::vector<float> values, bool train)
    : shape(std::move(s)), data(std::move(values)), requires_grad(train) {
  require(elements(shape) == data.size(), "Tensor data size mismatch");
  if (train)
    grad.assign(data.size(), 0);
}
T tensor(Shape s, std::vector<float> v, bool train) {
  return std::make_shared<Tensor>(std::move(s), std::move(v), train);
}
T zeros(Shape s, bool train) {
  auto n = elements(s);
  return tensor(std::move(s), std::vector<float>(n, 0), train);
}
void backward(const T &loss) {
  require(loss->data.size() == 1 && loss->requires_grad,
          "Backward requires differentiable scalar");
  std::vector<T> order;
  std::unordered_set<Tensor *> seen;
  std::function<void(const T &)> visit = [&](const T &t) {
    if (!seen.insert(t.get()).second)
      return;
    for (const auto &p : t->parents)
      visit(p);
    order.push_back(t);
  };
  visit(loss);
  for (auto &t : order)
    if (t->requires_grad)
      std::fill(t->grad.begin(), t->grad.end(), 0);
  loss->grad[0] = 1;
  for (auto it = order.rbegin(); it != order.rend(); ++it)
    if ((*it)->vjp)
      (*it)->vjp((*it)->grad);
}
T add(const T &a, const T &b) { return binary(a, b, false); }
T mul(const T &a, const T &b) { return binary(a, b, true); }
T scale(const T &a, float s) { return mul(a, tensor({1}, {s})); }
T reshape(const T &a, Shape s) {
  require(elements(s) == a->data.size(), "Reshape element mismatch");
  auto o = operation(std::move(s), {a});
  o->data = a->data;
  if (o->requires_grad)
    o->vjp = [a](const auto &g) {
      for (std::size_t i = 0; i < g.size(); ++i)
        accumulate(a, i, g[i]);
    };
  return o;
}
T permute(const T &a, Shape axes) {
  require(axes.size() == a->shape.size(), "Permutation rank mismatch");
  auto check = axes;
  std::sort(check.begin(), check.end());
  for (std::size_t i = 0; i < check.size(); ++i)
    require(check[i] == i, "Invalid permutation");
  Shape s;
  for (auto axis : axes)
    s.push_back(a->shape[axis]);
  auto st = strides(a->shape);
  auto o = operation(s, {a});
  auto map = std::make_shared<std::vector<std::size_t>>(o->data.size());
  for (std::size_t i = 0; i < o->data.size(); ++i) {
    auto rem = i, src = std::size_t(0);
    for (std::size_t j = s.size(); j > 0; --j) {
      src += (rem % s[j - 1]) * st[axes[j - 1]];
      rem /= s[j - 1];
    }
    (*map)[i] = src;
    o->data[i] = a->data[src];
  }
  if (o->requires_grad)
    o->vjp = [a, map](const auto &g) {
      for (std::size_t i = 0; i < g.size(); ++i)
        accumulate(a, (*map)[i], g[i]);
    };
  return o;
}
T cat(const std::vector<T> &ts, std::size_t axis) {
  require(!ts.empty() && axis < ts[0]->shape.size(), "Invalid concatenate");
  auto s = ts[0]->shape;
  s[axis] = 0;
  for (auto &t : ts) {
    require(t->shape.size() == s.size(), "Concatenate rank mismatch");
    for (std::size_t k = 0; k < s.size(); ++k)
      if (k != axis)
        require(s[k] == t->shape[k], "Concatenate extent mismatch");
    s[axis] += t->shape[axis];
  }
  auto inner = std::size_t(1), outer = std::size_t(1);
  for (std::size_t k = axis + 1; k < s.size(); ++k)
    inner *= s[k];
  for (std::size_t k = 0; k < axis; ++k)
    outer *= s[k];
  auto o = operation(s, ts);
  for (std::size_t b = 0; b < outer; ++b) {
    auto dst = b * s[axis] * inner;
    for (auto &t : ts) {
      auto count = t->shape[axis] * inner;
      std::copy_n(t->data.begin() + b * count, count, o->data.begin() + dst);
      dst += count;
    }
  }
  if (o->requires_grad)
    o->vjp = [ts, axis, inner, outer, s](const auto &g) {
      for (std::size_t b = 0; b < outer; ++b) {
        auto src = b * s[axis] * inner;
        for (auto &t : ts) {
          auto count = t->shape[axis] * inner;
          for (std::size_t i = 0; i < count; ++i)
            accumulate(t, b * count + i, g[src + i]);
          src += count;
        }
      }
    };
  return o;
}
T matmul(const T &a, const T &b) {
  require(a->shape.size() == 3 && b->shape.size() == 3,
          "Matmul expects rank three");
  auto B = std::max(a->shape[0], b->shape[0]), M = a->shape[1], K = a->shape[2],
       N = b->shape[2];
  require(K == b->shape[1] && (a->shape[0] == B || a->shape[0] == 1) &&
              (b->shape[0] == B || b->shape[0] == 1),
          "Matmul extent mismatch");
  auto o = operation({B, M, N}, {a, b});
#ifdef OCUWEAVE_CUDA
  if (cuda_enabled())
    cuda_matmul_forward(a->data.data(), b->data.data(), o->data.data(), B,
                        a->shape[0], b->shape[0], M, K, N);
  else
#endif
    for (std::size_t q = 0; q < B; ++q)
      for (std::size_t i = 0; i < M; ++i)
        for (std::size_t k = 0; k < K; ++k) {
          float av = a->data[(q % a->shape[0] * M + i) * K + k];
          for (std::size_t j = 0; j < N; ++j)
            o->data[(q * M + i) * N + j] +=
                av * b->data[(q % b->shape[0] * K + k) * N + j];
        }
  if (o->requires_grad)
    o->vjp = [a, b, B, M, K, N](const auto &g) {
#ifdef OCUWEAVE_CUDA
      if (cuda_enabled()) {
        std::vector<float> da(a->data.size()), db(b->data.size());
        cuda_matmul_backward(a->data.data(), b->data.data(), g.data(),
                             da.data(), db.data(), B, a->shape[0], b->shape[0],
                             M, K, N);
        for (std::size_t i = 0; i < da.size(); ++i)
          accumulate(a, i, da[i]);
        for (std::size_t i = 0; i < db.size(); ++i)
          accumulate(b, i, db[i]);
        return;
      }
#endif
      for (std::size_t q = 0; q < B; ++q)
        for (std::size_t i = 0; i < M; ++i)
          for (std::size_t k = 0; k < K; ++k)
            for (std::size_t j = 0; j < N; ++j) {
              auto ai = (q % a->shape[0] * M + i) * K + k,
                   bi = (q % b->shape[0] * K + k) * N + j;
              float z = g[(q * M + i) * N + j];
              accumulate(a, ai, z * b->data[bi]);
              accumulate(b, bi, z * a->data[ai]);
            }
    };
  return o;
}
T conv2d(const T &x, const T &w, const T &b, int stride, int padding,
         int groups) {
  require(x->shape.size() == 4 && w->shape.size() == 4,
          "Convolution expects NCHW/OIHW");
  require(stride > 0 && padding >= 0 && groups > 0,
          "Invalid convolution settings");
  int B = x->shape[0], C = x->shape[1], H = x->shape[2], W = x->shape[3],
      O = w->shape[0], KH = w->shape[2], KW = w->shape[3];
  require(C % groups == 0 && O % groups == 0 &&
              w->shape[1] == std::size_t(C / groups) &&
              b->data.size() == std::size_t(O),
          "Convolution channel mismatch");
  require(H + 2 * padding >= KH && W + 2 * padding >= KW,
          "Convolution kernel exceeds input");
  int OH = (H + 2 * padding - KH) / stride + 1,
      OW = (W + 2 * padding - KW) / stride + 1;
  auto o = operation(
      {std::size_t(B), std::size_t(O), std::size_t(OH), std::size_t(OW)},
      {x, w, b});
#ifdef OCUWEAVE_CUDA
  if (cuda_enabled())
    cuda_conv_forward(x->data.data(), w->data.data(), b->data.data(),
                      o->data.data(), B, C, H, W, O, KH, KW, stride, padding,
                      groups);
  else
#endif
    for (int n = 0; n < B; ++n)
      for (int oc = 0; oc < O; ++oc)
        for (int y = 0; y < OH; ++y)
          for (int z = 0; z < OW; ++z) {
            float v = b->data[oc];
            for (int ci = 0; ci < C / groups; ++ci)
              for (int ky = 0; ky < KH; ++ky)
                for (int kx = 0; kx < KW; ++kx) {
                  int iy = y * stride + ky - padding,
                      ix = z * stride + kx - padding;
                  if (iy < 0 || iy >= H || ix < 0 || ix >= W)
                    continue;
                  int ic = oc / (O / groups) * (C / groups) + ci;
                  v += x->data[((n * C + ic) * H + iy) * W + ix] *
                       w->data[((oc * (C / groups) + ci) * KH + ky) * KW + kx];
                }
            o->data[((n * O + oc) * OH + y) * OW + z] = v;
          }
  if (o->requires_grad)
    o->vjp = [=](const auto &g) {
#ifdef OCUWEAVE_CUDA
      if (cuda_enabled()) {
        std::vector<float> dx(x->data.size()), dw(w->data.size()),
            db(b->data.size());
        cuda_conv_backward(x->data.data(), w->data.data(), g.data(), dx.data(),
                           dw.data(), db.data(), B, C, H, W, O, KH, KW, stride,
                           padding, groups);
        for (std::size_t i = 0; i < dx.size(); ++i)
          accumulate(x, i, dx[i]);
        for (std::size_t i = 0; i < dw.size(); ++i)
          accumulate(w, i, dw[i]);
        for (std::size_t i = 0; i < db.size(); ++i)
          accumulate(b, i, db[i]);
        return;
      }
#endif
      for (int n = 0; n < B; ++n)
        for (int oc = 0; oc < O; ++oc)
          for (int y = 0; y < OH; ++y)
            for (int z = 0; z < OW; ++z) {
              float v = g[((n * O + oc) * OH + y) * OW + z];
              accumulate(b, oc, v);
              for (int ci = 0; ci < C / groups; ++ci)
                for (int ky = 0; ky < KH; ++ky)
                  for (int kx = 0; kx < KW; ++kx) {
                    int iy = y * stride + ky - padding,
                        ix = z * stride + kx - padding;
                    if (iy < 0 || iy >= H || ix < 0 || ix >= W)
                      continue;
                    int ic = oc / (O / groups) * (C / groups) + ci;
                    auto xi = ((n * C + ic) * H + iy) * W + ix,
                         wi = ((oc * (C / groups) + ci) * KH + ky) * KW + kx;
                    accumulate(x, xi, v * w->data[wi]);
                    accumulate(w, wi, v * x->data[xi]);
                  }
            }
    };
  return o;
}
T group_norm(const T &x, const T &w, const T &b, int groups, float eps) {
  require(x->shape.size() == 4 && groups > 0 && eps > 0,
          "Invalid normalization settings");
  auto B = x->shape[0], C = x->shape[1], HW = x->shape[2] * x->shape[3];
  require(C % groups == 0 && w->data.size() == C && b->data.size() == C,
          "Normalization channel mismatch");
  auto S = C / groups * HW;
  auto o = operation(x->shape, {x, w, b});
  auto normalized = std::make_shared<std::vector<float>>(x->data.size());
  auto inv = std::make_shared<std::vector<float>>(B * groups);
  for (std::size_t q = 0; q < B * groups; ++q) {
    double mean = 0, var = 0;
    for (std::size_t i = 0; i < S; ++i)
      mean += x->data[q * S + i];
    mean /= S;
    for (std::size_t i = 0; i < S; ++i) {
      double d = x->data[q * S + i] - mean;
      var += d * d;
    }
    float iv = 1 / std::sqrt(var / S + eps);
    (*inv)[q] = iv;
    for (std::size_t i = 0; i < S; ++i) {
      auto k = q * S + i, c = k / HW % C;
      (*normalized)[k] = (x->data[k] - mean) * iv;
      o->data[k] = (*normalized)[k] * w->data[c] + b->data[c];
    }
  }
  if (o->requires_grad)
    o->vjp = [=](const auto &g) {
      for (std::size_t q = 0; q < B * groups; ++q) {
        double sum = 0, dot = 0;
        for (std::size_t i = 0; i < S; ++i) {
          auto k = q * S + i, c = k / HW % C;
          float dy = g[k] * w->data[c];
          sum += dy;
          dot += dy * (*normalized)[k];
          accumulate(w, c, g[k] * (*normalized)[k]);
          accumulate(b, c, g[k]);
        }
        for (std::size_t i = 0; i < S; ++i) {
          auto k = q * S + i, c = k / HW % C;
          accumulate(x, k,
                     (*inv)[q] * (g[k] * w->data[c] - sum / S -
                                  (*normalized)[k] * dot / S));
        }
      }
    };
  return o;
}
T gelu(const T &x) {
  auto o = operation(x->shape, {x});
  constexpr double rt2 = 1.4142135623730951;
  for (std::size_t i = 0; i < x->data.size(); ++i)
    o->data[i] = .5 * x->data[i] * (1 + std::erf(x->data[i] / rt2));
  if (o->requires_grad)
    o->vjp = [x](const auto &g) {
      for (std::size_t i = 0; i < g.size(); ++i) {
        double v = x->data[i];
        accumulate(x, i,
                   g[i] * (.5 * (1 + std::erf(v / 1.4142135623730951)) +
                           v * std::exp(-.5 * v * v) / 2.5066282746310005));
      }
    };
  return o;
}
T sigmoid(const T &x) {
  auto o = operation(x->shape, {x});
  for (std::size_t i = 0; i < x->data.size(); ++i)
    o->data[i] = logistic(x->data[i]);
  if (o->requires_grad) {
    auto values = o->data;
    o->vjp = [x, values](const auto &g) {
      for (std::size_t i = 0; i < g.size(); ++i)
        accumulate(x, i, g[i] * values[i] * (1 - values[i]));
    };
  }
  return o;
}
T softmax(const T &x, std::size_t axis) {
  require(axis < x->shape.size(), "Invalid softmax axis");
  auto dim = x->shape[axis], inner = std::size_t(1);
  for (std::size_t k = axis + 1; k < x->shape.size(); ++k)
    inner *= x->shape[k];
  auto outer = x->data.size() / dim / inner;
  auto o = operation(x->shape, {x});
  for (std::size_t b = 0; b < outer; ++b)
    for (std::size_t i = 0; i < inner; ++i) {
      float mx = -std::numeric_limits<float>::infinity();
      for (std::size_t d = 0; d < dim; ++d)
        mx = std::max(mx, x->data[(b * dim + d) * inner + i]);
      double total = 0;
      for (std::size_t d = 0; d < dim; ++d) {
        auto k = (b * dim + d) * inner + i;
        o->data[k] = std::exp(x->data[k] - mx);
        total += o->data[k];
      }
      for (std::size_t d = 0; d < dim; ++d)
        o->data[(b * dim + d) * inner + i] /= total;
    }
  if (o->requires_grad) {
    auto values = o->data;
    o->vjp = [=](const auto &g) {
      for (std::size_t b = 0; b < outer; ++b)
        for (std::size_t i = 0; i < inner; ++i) {
          double dot = 0;
          for (std::size_t d = 0; d < dim; ++d) {
            auto k = (b * dim + d) * inner + i;
            dot += g[k] * values[k];
          }
          for (std::size_t d = 0; d < dim; ++d) {
            auto k = (b * dim + d) * inner + i;
            accumulate(x, k, values[k] * (g[k] - dot));
          }
        }
    };
  }
  return o;
}
T resize(const T &x, std::size_t H, std::size_t W) {
  require(x->shape.size() == 4 && H > 0 && W > 0,
          "Resize expects NCHW and positive extents");
  auto ih = x->shape[2], iw = x->shape[3], planes = x->shape[0] * x->shape[1];
  auto o = operation({x->shape[0], x->shape[1], H, W}, {x});
  auto apply = [=](const std::vector<float> *g, std::vector<float> *out) {
    for (std::size_t n = 0; n < planes; ++n)
      for (std::size_t y = 0; y < H; ++y)
        for (std::size_t z = 0; z < W; ++z) {
          double fy = std::max(0., (y + .5) * ih / H - .5),
                 fx = std::max(0., (z + .5) * iw / W - .5);
          auto y0 = std::min(ih - 1, std::size_t(fy)),
               x0 = std::min(iw - 1, std::size_t(fx)),
               y1 = std::min(ih - 1, y0 + 1), x1 = std::min(iw - 1, x0 + 1);
          float dy = fy - y0, dx = fx - x0;
          std::size_t indices[4] = {
              (n * ih + y0) * iw + x0, (n * ih + y0) * iw + x1,
              (n * ih + y1) * iw + x0, (n * ih + y1) * iw + x1};
          float weights[4] = {(1 - dy) * (1 - dx), (1 - dy) * dx, dy * (1 - dx),
                              dy * dx};
          auto k = (n * H + y) * W + z;
          for (int j = 0; j < 4; ++j) {
            if (g)
              accumulate(x, indices[j], (*g)[k] * weights[j]);
            else
              (*out)[k] += x->data[indices[j]] * weights[j];
          }
        }
  };
  apply(nullptr, &o->data);
  if (o->requires_grad)
    o->vjp = [apply](const auto &g) { apply(&g, nullptr); };
  return o;
}
T reduce_sum(const T &x, std::size_t axis, bool keep) {
  require(axis < x->shape.size(), "Invalid reduction axis");
  auto s = x->shape;
  auto d = s[axis];
  s[axis] = 1;
  if (!keep)
    s.erase(s.begin() + axis);
  if (s.empty())
    s = {1};
  auto inner = std::size_t(1);
  for (std::size_t k = axis + 1; k < x->shape.size(); ++k)
    inner *= x->shape[k];
  auto outer = x->data.size() / d / inner;
  auto o = operation(s, {x});
  for (std::size_t b = 0; b < outer; ++b)
    for (std::size_t c = 0; c < d; ++c)
      for (std::size_t i = 0; i < inner; ++i)
        o->data[b * inner + i] += x->data[(b * d + c) * inner + i];
  if (o->requires_grad)
    o->vjp = [=](const auto &g) {
      for (std::size_t b = 0; b < outer; ++b)
        for (std::size_t c = 0; c < d; ++c)
          for (std::size_t i = 0; i < inner; ++i)
            accumulate(x, (b * d + c) * inner + i, g[b * inner + i]);
    };
  return o;
}
T spatial_gate(const T &m, const T &w) {
  return reduce_sum(mul(m, w), 1, true);
}
LossResult observed_loss(const T &z, const T &y, const T &valid,
                         const std::vector<unsigned char> &av,
                         const std::vector<float> &pw) {
  require(z->shape == y->shape && z->shape.size() == 4,
          "Loss expects matching NCHW");
  auto B = z->shape[0], C = z->shape[1], S = z->shape[2] * z->shape[3];
  require(valid->shape == Shape({B, 1, z->shape[2], z->shape[3]}) &&
              av.size() == B * C,
          "Invalid loss availability/validity");
  require(pw.empty() || pw.size() == C, "Positive weight size mismatch");
  for (float w : pw)
    require(std::isfinite(w) && w > 0, "Positive finite weights required");
  auto o = operation({1}, {z});
  auto dz = std::make_shared<std::vector<float>>(z->data.size(), 0);
  std::size_t pixels = 0, pairs = 0;
  double bce = 0, dice = 0;
  for (std::size_t n = 0; n < B; ++n)
    for (std::size_t c = 0; c < C; ++c) {
      require(av[n * C + c] <= 1, "Availability must be binary");
      if (!av[n * C + c])
        continue;
      std::size_t count = 0;
      double ps = 0, ys = 0, py = 0;
      float weight = pw.empty() ? 1 : pw[c];
      for (std::size_t i = 0; i < S; ++i) {
        auto vi = n * S + i, k = (n * C + c) * S + i;
        require(valid->data[vi] == 0 || valid->data[vi] == 1,
                "Validity must be binary");
        if (!valid->data[vi])
          continue;
        float a = z->data[k], t = y->data[k];
        require(std::isfinite(a) && std::isfinite(t) && t >= 0 && t <= 1,
                "Finite supervised logits/targets in [0,1] required");
        float p = logistic(a);
        ps += p;
        ys += t;
        py += p * t;
        bce += (1 - t) * std::max(a, 0.f) + weight * t * std::max(-a, 0.f) +
               (1 - t + weight * t) * std::log1p(std::exp(-std::abs(a)));
        ++count;
      }
      require(count > 0, "Observed pair has no valid pixels");
      pixels += count;
      ++pairs;
      double numerator = 2 * py + 1, denominator = ps + ys + 1;
      dice += 1 - numerator / denominator;
      for (std::size_t i = 0; i < S; ++i) {
        auto k = (n * C + c) * S + i;
        if (!valid->data[n * S + i])
          continue;
        float p = logistic(z->data[k]), t = y->data[k];
        (*dz)[k] = (numerator - 2 * t * denominator) /
                   (denominator * denominator) * p * (1 - p);
      }
    }
  require(pairs > 0, "No observed labels");
  for (std::size_t n = 0; n < B; ++n)
    for (std::size_t c = 0; c < C; ++c)
      if (av[n * C + c])
        for (std::size_t i = 0; i < S; ++i) {
          auto k = (n * C + c) * S + i;
          if (!valid->data[n * S + i])
            continue;
          float p = logistic(z->data[k]), t = y->data[k],
                w = pw.empty() ? 1 : pw[c];
          (*dz)[k] =
              (*dz)[k] / pairs + ((1 - t) * p + w * t * (p - 1)) / pixels;
        }
  float bc = bce / pixels, dc = dice / pairs;
  o->data[0] = bc + dc;
  if (o->requires_grad)
    o->vjp = [z, dz](const auto &g) {
      for (std::size_t i = 0; i < dz->size(); ++i)
        accumulate(z, i, g[0] * (*dz)[i]);
    };
  return {o, bc, dc};
}
} // namespace ocuweave::nn
