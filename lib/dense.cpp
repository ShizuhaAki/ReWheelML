#include "dense.hpp"

#include <string>

#include "functional.hpp"
#include "ndarray.hpp"

Dense::Dense(int input_dim, int output_dim, string activation) {
    W = NDArray({input_dim, output_dim});
    b = NDArray({1, output_dim});
    dW = NDArray({input_dim, output_dim});
    db = NDArray({1, output_dim});
    if (activation == "sigmoid") {
        activation_fn = sigmoid;
        grad_fn = sigmoid_grad;
    } else if (activation == "linear") {
        activation_fn = linear;
        grad_fn = linear_grad;
    } else if (activation == "relu") {
        activation_fn = relu;
        grad_fn = relu_grad;
    } else if (activation == "tanh") {
        activation_fn = tanh_fwd;
        grad_fn = tanh_grad;
    }
    W.init_weights();
    b.init_weights();
}

NDArray Dense::forward(const NDArray& x) {
    input_cache = x;
    auto Z = x.matmul(W) + b;
    output_cache = activation_fn(Z);  // cache y
    return output_cache;
}
NDArray Dense::backward(const NDArray& dY) {
    auto dZ = grad_fn(output_cache, dY);
    dW = input_cache.transpose2d().matmul(dZ);
    db = dZ.sum(0, true);
    auto dX = dZ.matmul(W.transpose2d());
    return dX;
}

void Dense::step(double lr) {
    W = W - dW * lr;
    b = b - db * lr;
}