#include "module/dense.hpp"

#include <stdexcept>
#include <string>

#include "functional.hpp"
#include "ndarray.hpp"

static void configure_activation(const string& activation,
                                 ForwardFunc& activation_fn,
                                 GradFunc& grad_fn) {
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
    } else {
        throw invalid_argument("unknown activation: " + activation);
    }
}
Linear::Linear(int input_dim, int output_dim) {
    W = NDArray({input_dim, output_dim});
    b = NDArray({1, output_dim});
    dW = NDArray({input_dim, output_dim});
    db = NDArray({1, output_dim});
    W.init_weights();
    b.init_weights();
}

NDArray Linear::forward(const NDArray& x) {
    input_cache = x;
    return x.matmul(W) + b;
}

NDArray Linear::backward(const NDArray& grad_out) {
    dW = input_cache.transpose2d().matmul(grad_out);
    auto grad_out_copy = grad_out;
    db = grad_out_copy.sum(0, true);
    auto dX = grad_out.matmul(W.transpose2d());
    return dX;
}

void Linear::step(float lr) {
    W = W - dW * lr;
    b = b - db * lr;
}

std::vector<Parameter> Linear::parameters() {
    return {{&W, &dW}, {&b, &db}};
}

Activation::Activation(string activation) {
    configure_activation(activation, activation_fn, grad_fn);
}

NDArray Activation::forward(const NDArray& x) {
    output_cache = activation_fn(x);
    return output_cache;
}

NDArray Activation::backward(const NDArray& grad_out) {
    return grad_fn(output_cache, grad_out);
}

void Activation::step(float /*lr*/) {}
