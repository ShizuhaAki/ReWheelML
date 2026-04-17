#pragma once

#include "module/module.hpp"

class BatchNorm2D : public Module {
   private:
    int channels;
    float eps;
    float momentum;
    bool training;

    NDArray gamma;
    NDArray beta;
    NDArray dgamma;
    NDArray dbeta;
    NDArray running_mean;
    NDArray running_var;

    NDArray x_centered_cache;
    NDArray x_hat_cache;
    NDArray inv_std_cache;

   public:
    BatchNorm2D(int channels, float eps = 1e-5f, float momentum = 0.1f);

    NDArray forward(const NDArray& x) override;
    NDArray backward(const NDArray& grad_out) override;
    void step(float lr) override;
    void train() override;
    void eval() override;
    std::vector<Parameter> parameters() override;
};
