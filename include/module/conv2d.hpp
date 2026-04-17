#pragma once

#include "module/module.hpp"
#include "ndarray.hpp"
class Conv2D : public Module {
   private:
    int in_channels, out_channels, kernel_size, stride, padding;
    NDArray W, b, dW, db, input_cache, col_cache;

   public:
    Conv2D(int in_channels, int out_channels, int kernel_size, int stride = 1,
           int padding = 0);
    NDArray forward(const NDArray& x) override;
    NDArray backward(const NDArray& grad_out) override;
    void step(float lr) override;
    std::vector<Parameter> parameters() override;
};
