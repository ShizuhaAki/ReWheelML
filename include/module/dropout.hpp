#pragma once

#include <random>

#include "module.hpp"

class Dropout : public Module {
   private:
    float p;
    float keep_prob;
    bool training;
    NDArray mask;
    std::mt19937 rng;
    std::bernoulli_distribution keep_dist;

   public:
    explicit Dropout(float p, unsigned int seed = 42);
    NDArray forward(const NDArray& x) override;
    NDArray backward(const NDArray& grad_out) override;
    void step(float lr) override;
    void train() override;
    void eval() override;
};
