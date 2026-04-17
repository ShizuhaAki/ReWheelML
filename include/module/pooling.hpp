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
    void step(float /*lr*/) override {}
};
