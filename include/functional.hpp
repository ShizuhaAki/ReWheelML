#pragma once

#include <cmath>

#include "ndarray.hpp"

inline NDArray relu(const NDArray& x) {
    return x.for_each_element([](double v) { return v > 0 ? v : 0; });
}
inline NDArray sigmoid(const NDArray& x) {
    return x.for_each_element([](double v) { return 1.0 / (1.0 + exp(-v)); });
}
inline NDArray relu_grad(const NDArray& x, const NDArray& grad_out) {
    return x.for_each_element_indexed(
        [&grad_out](double v, int idx) { return v > 0 ? grad_out[idx] : 0; });
}

inline NDArray mean_squared_error(const NDArray& pred, const NDArray& target) {
    auto diff = pred - target;
    return diff * diff;
}