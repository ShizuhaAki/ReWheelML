#pragma once

#include "functional.hpp"
#include "module.hpp"
#include "ndarray.hpp"
class Dense : public Module {
   private:
    NDArray W, b, dW, db, input_cache, output_cache;
    decltype(&relu_grad) grad_fn;

   public:
    Dense(int input_dim, int output_dim);
    NDArray forward(const NDArray& x);
    NDArray backward(const NDArray& dY);
    void step(double lr);
};