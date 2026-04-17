#pragma once

#include <cmath>
#include <utility>
#include <vector>

#include "optim/optim.hpp"

class Adam : public Optimizer {
   private:
    float lr;
    float beta1;
    float beta2;
    float eps;
    int t;
    std::vector<NDArray> m;
    std::vector<NDArray> v;

   public:
    Adam(std::vector<Parameter> params, float lr = 1e-3f, float beta1 = 0.9f,
         float beta2 = 0.999f, float eps = 1e-8f)
        : Optimizer(std::move(params)),
          lr(lr),
          beta1(beta1),
          beta2(beta2),
          eps(eps),
          t(0) {
        m.reserve(this->params.size());
        v.reserve(this->params.size());
        for (const auto& param : this->params) {
            m.emplace_back(param.value->shape);
            m.back().fill(0.0f);
            v.emplace_back(param.value->shape);
            v.back().fill(0.0f);
        }
    }

    void step() override {
        ++t;
        float bias_correction1 = 1.0f - std::pow(beta1, static_cast<float>(t));
        float bias_correction2 = 1.0f - std::pow(beta2, static_cast<float>(t));

        for (int i = 0; i < static_cast<int>(params.size()); ++i) {
            auto* value = params[i].value;
            auto* grad = params[i].grad;
            if (grad == nullptr) {
                continue;
            }
            for (int j = 0; j < value->numel; ++j) {
                float g = (*grad)[j];
                m[i][j] = beta1 * m[i][j] + (1.0f - beta1) * g;
                v[i][j] = beta2 * v[i][j] + (1.0f - beta2) * g * g;

                float m_hat = m[i][j] / bias_correction1;
                float v_hat = v[i][j] / bias_correction2;
                (*value)[j] -= lr * m_hat / (std::sqrt(v_hat) + eps);
            }
        }
    }

    float get_lr() const override { return lr; }

    void set_lr(float new_lr) override { lr = new_lr; }
};
