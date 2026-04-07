#pragma once

#include <cmath>

#include "ndarray.hpp"

inline NDArray relu(const NDArray& x) {
    return x.for_each_element([](float v) { return v > 0 ? v : 0; });
}
inline NDArray sigmoid(const NDArray& x) {
    return x.for_each_element([](float v) { return 1.0 / (1.0 + exp(-v)); });
}
inline NDArray linear(const NDArray& x) { return x; }

inline NDArray linear_grad(const NDArray& /*y*/, const NDArray& grad_out) {
    return grad_out;
}

inline NDArray relu_grad(const NDArray& x, const NDArray& grad_out) {
    return x.for_each_element_indexed(
        [&grad_out](float v, int idx) { return v > 0 ? grad_out[idx] : 0; });
}
inline NDArray sigmoid_grad(const NDArray& y, const NDArray& grad_out) {
    return y.for_each_element_indexed(
        [&](float v, int idx) { return grad_out[idx] * v * (1.0 - v); });
}

inline NDArray tanh_fwd(const NDArray& x) {
    return x.for_each_element([](float v) { return std::tanh(v); });
}

inline NDArray tanh_grad(const NDArray& y, const NDArray& grad_out) {
    return y.for_each_element_indexed(
        [&](float v, int idx) { return grad_out[idx] * (1.0 - v * v); });
}

// non-grad functions begins here
inline NDArray squared_error(const NDArray& pred, const NDArray& target) {
    auto diff = pred - target;
    return diff * diff;
}

inline NDArray softmax(const NDArray& logits) {
    NDArray probs(logits.shape);
    int batch = logits.shape[0];
    int classes = logits.shape[1];

    for (int i = 0; i < batch; i++) {
        auto row_max = logits.get({i, 0});
        for (int j = 1; j < classes; j++) {
            row_max = max(row_max, logits.get({i, j}));
        }

        auto sum_exp = 0.0f;
        for (int j = 0; j < classes; j++) {
            float v = exp(logits.get({i, j}) - row_max);
            probs.set({i, j}, v);
            sum_exp += v;
        }

        for (int j = 0; j < classes; j++) {
            probs.set({i, j}, probs.get({i, j}) / sum_exp);
        }
    }
    return probs;
}

inline float cross_entropy(const NDArray& probs, const NDArray& target) {
    const float eps = 1e-12f;
    int batch = probs.shape[0];
    int classes = probs.shape[1];

    float loss = 0.0f;
    for (int i = 0; i < batch; i++) {
        for (int j = 0; j < classes; j++) {
            float y = target.get({i, j});
            if (y > 0.0f) {
                float p = probs.get({i, j});
                loss -= y * log(p + eps);
            }
        }
    }
    return loss / static_cast<float>(batch);
}

inline NDArray cross_entropy_grad(const NDArray& probs, const NDArray& target) {
    int batch = probs.shape[0];
    return (probs - target) / static_cast<float>(batch);
}

// typedef goes to the bottom
typedef decltype(&relu_grad) GradFunc;
typedef decltype(&relu) ForwardFunc;