#include "module/dropout.hpp"

#include <stdexcept>

Dropout::Dropout(float p, unsigned int seed)
    : p(p),
      keep_prob(1.0f - p),
      training(true),
      rng(seed),
      keep_dist(static_cast<double>(keep_prob)) {
    if (p < 0.0f || p >= 1.0f) {
        throw std::invalid_argument("dropout probability must be in [0, 1)");
    }
}

NDArray Dropout::forward(const NDArray& x) {
    if (!training || p == 0.0f) {
        mask = NDArray();
        return x;
    }

    mask = NDArray(x.shape);
    for (int i = 0; i < x.numel; ++i) {
        mask[i] = keep_dist(rng) ? 1.0f / keep_prob : 0.0f;
    }
    return x * mask;
}

NDArray Dropout::backward(const NDArray& grad_out) {
    if (!training || p == 0.0f || mask.numel == 0) {
        return grad_out;
    }
    return grad_out * mask;
}

void Dropout::step(float /*lr*/) {}

void Dropout::train() { training = true; }

void Dropout::eval() { training = false; }
