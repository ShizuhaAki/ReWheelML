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
    virtual float get_lr() const = 0;
    virtual void set_lr(float lr) = 0;
    virtual void zero_grad() {
        for (auto& param : params) {
            if (param.grad != nullptr) {
                param.grad->fill(0.0f);
            }
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
            if (param.grad != nullptr) {
                *param.value = *param.value - (*param.grad * lr);
            }
        }
    }

    float get_lr() const override { return lr; }

    void set_lr(float new_lr) override { lr = new_lr; }
};
