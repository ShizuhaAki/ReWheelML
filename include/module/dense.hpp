#pragma once

#include <string>

#include "functional.hpp"
#include "module.hpp"
#include "ndarray.hpp"

class Linear : public Module {
   private:
    NDArray W, b, dW, db, input_cache;

   public:
    Linear(int input_dim, int output_dim);
    NDArray forward(const NDArray& x) override;
    NDArray backward(const NDArray& grad_out) override;
    void step(float lr) override;
    std::vector<Parameter> parameters() override;
};

class Activation : public Module {
   private:
    NDArray output_cache;
    GradFunc grad_fn;
    ForwardFunc activation_fn;

   public:
    explicit Activation(string activation = "sigmoid");
    NDArray forward(const NDArray& x);
    NDArray backward(const NDArray& grad_out);
    void step(float lr);
};
