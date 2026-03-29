#include "dense.hpp"

#include "functional.hpp"
#include "ndarray.hpp"

Dense::Dense(int input_dim, int output_dim) {
    W = NDArray({input_dim, output_dim});
    b = NDArray({1, output_dim});
    dW = NDArray({input_dim, output_dim});
    db = NDArray({1, output_dim});
    W.fill(0);
    b.fill(0);

    grad_fn = relu_grad;
}

NDArray Dense::forward(const NDArray& x) {
    input_cache = x;
    auto Z = x.matmul(W) + b;
    output_cache = relu(Z);
    return output_cache;
}

NDArray Dense::backward(const NDArray& dY) {
    auto dZ = grad_fn(output_cache, dY);
    auto dW = input_cache.transpose2d().matmul(dZ);
    auto db = dZ.sum(0, true);
    auto dX = dZ.matmul(W.transpose2d());
    return dX;
}

void Dense::step(double lr) {
    W = W - dW * lr;
    b = b - db * lr;
}