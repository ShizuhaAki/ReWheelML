#pragma once

#include <utility>
#include <vector>

#include "module/module.hpp"

class Optimizer {
   protected:
    std::vector<Parameter> params;

   public:
    explicit Optimizer(std::vector<Parameter> params)
        : params(std::move(params)) {}
    virtual ~Optimizer() = default;
    virtual void step() = 0;
    virtual void zero_grad() {
        for (auto& param : params) {
            param.grad->fill(0.0f);
        }
    }
};

class SGD : public Optimizer {
   private:
    float lr;

   public:
    SGD(std::vector<Parameter> params, float lr)
        : Optimizer(std::move(params)), lr(lr) {}

    void step() override {
        for (auto& param : params) {
            *param.value = *param.value - (*param.grad * lr);
        }
    }
};
