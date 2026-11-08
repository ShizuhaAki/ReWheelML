#pragma once

#include "module/module.hpp"
#include "ndarray.hpp"

class MaxPool2D : public Module {
   public:
    int pool_size;
    int stride;

    NDArray input_cache;
    NDArray max_h_cache;
    NDArray max_w_cache;

    MaxPool2D(int pool_size, int stride);

    NDArray forward(const NDArray& x) override;
    NDArray backward(const NDArray& grad_out) override;

    /* @doc
    ## Optimizer Step

    Max pooling has no weights or biases. `step` is therefore a no-op: the
    layer participates in forward and backward propagation, but there are no
    parameters for an optimizer to update.
    */
    void step(float /*lr*/) override {}
};
