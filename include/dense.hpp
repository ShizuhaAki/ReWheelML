#pragma once

#include <string>

#include "functional.hpp"
#include "module.hpp"
#include "ndarray.hpp"
class Dense : public Module {
   private:
    NDArray W, b, dW, db, input_cache, output_cache;
    decltype(&relu_grad) grad_fn;
    decltype(&relu) activation_fn;

   public:
    Dense(int input_dim, int output_dim, string activation = "sigmoid");
    NDArray forward(const NDArray& x);
    NDArray backward(const NDArray& dY);
    void step(double lr);
};