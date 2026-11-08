#pragma once

#include <cmath>
#include <utility>
#include <vector>

#include "optim/optim.hpp"

/* @doc
# Adam Optimizer

This header implements Adam as an `Optimizer` subclass. Adam updates the same
`Parameter` objects as SGD, but it keeps extra state for every scalar parameter:
a smoothed gradient and a smoothed squared gradient.

When reading the code below, keep one scalar parameter $\theta$ in mind. At
training step $t$, its current gradient is $g_t$. Adam maintains

$$
m_t \approx \text{recent average of } g_t,\qquad
v_t \approx \text{recent average of } g_t^2.
$$

The first value tells the optimizer the direction that gradients have been
pointing recently. The second value estimates how large gradients usually are
for this specific parameter. Dividing by $\sqrt{v_t}$ gives each parameter its
own effective learning rate.
*/

class Adam : public Optimizer {
   private:
    /* @doc
    ## Optimizer State

    The base `Optimizer` stores `params`, the list of `(value, grad)` tensor
    pairs. Adam adds the hyperparameters and the running buffers needed by the
    update rule.

    - `lr` is the global learning-rate scale $\alpha$ 
    - `beta1` controls how slowly the first moment `m` changes.
    - `beta2` controls how slowly the second moment `v` changes.
    - `eps` prevents division by zero when `v` is very small.
    - `t` counts how many `step()` calls have happened.

    The vectors `m` and `v` are parallel to `params`: `m[i]` and `v[i]` store
    the Adam history for `params[i]`. Their shapes must match the parameter
    tensor so that every scalar weight gets its own moving averages.
    */

    float lr;
    float beta1;
    float beta2;
    float eps;
    int t;
    std::vector<NDArray> m;
    std::vector<NDArray> v;

   public:
    /* @doc
    ## Constructor

    The constructor first lets `Optimizer` take ownership of the parameter list.
    Then it stores the Adam hyperparameters. The default values are the standard
    Adam defaults:

    $$
    \alpha = 10^{-3},\qquad \beta_1 = 0.9,\qquad
    \beta_2 = 0.999,\qquad \epsilon = 10^{-8}.
    $$

    The important implementation detail is the loop over `this->params`. For
    every parameter tensor, the constructor creates two same-shaped tensors $m$
    and $v$. Both are initialized to zero. At this point Adam has not seen any
    gradients, so its first and second moment estimates are empty histories.
    */

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

    /* @doc
    ## One Optimization Step

    `step()` applies one Adam update to every parameter tensor. The first line
    increments `t`, because the bias-correction factors depend on the number of
    updates already performed:

    $$
    1-\beta_1^t,\qquad 1-\beta_2^t.
    $$

    These corrections matter most at the start of training. Since `m` and `v`
    begin at zero, their raw moving averages are biased toward zero during early
    steps. Dividing by the correction factors gives the estimates used in the
    original Adam algorithm:

    $$
    \hat{m}_t = \frac{m_t}{1-\beta_1^t},\qquad
    \hat{v}_t = \frac{v_t}{1-\beta_2^t}.
    $$
    */

    void step() override {
        ++t;
        float bias_correction1 = 1.0f - std::pow(beta1, static_cast<float>(t));
        float bias_correction2 = 1.0f - std::pow(beta2, static_cast<float>(t));

        /* @doc
        The outer loop walks through the parameter list exposed by the model.
        Each `Parameter` contains a pointer to the actual tensor and a pointer
        to its gradient tensor. If `grad` is null, this parameter has no
        gradient to apply, so Adam skips it exactly like SGD does.

        The optimizer never needs to know whether a tensor came from a dense
        layer, convolution layer, or batch-normalization layer. It only needs
        `value`, `grad`, and the matching entries in `m` and `v`.
        */

        for (int i = 0; i < static_cast<int>(params.size()); ++i) {
            auto* value = params[i].value;
            auto* grad = params[i].grad;
            if (grad == nullptr) {
                continue;
            }
            /* @doc
            Inside each tensor, Adam updates one scalar at a time. For scalar
            index `j`, the gradient is

            $$
            g_t = \texttt{(*grad)[j]}.
            $$

            The first moving average is a momentum-like estimate:

            $$
            m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t.
            $$

            If gradients keep pointing in the same direction, `m` keeps that
            direction and smooths out small batch-to-batch noise.
            */

            for (int j = 0; j < value->numel; ++j) {
                float g = (*grad)[j];
                m[i][j] = beta1 * m[i][j] + (1.0f - beta1) * g;

                /* @doc
                The second moving average tracks squared gradients:

                $$
                v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2.
                $$

                This is the adaptive part of Adam. A parameter that often
                receives large gradients will accumulate a larger `v`, which
                makes the denominator larger and therefore makes future updates
                smaller. A parameter with consistently small gradients receives
                a comparatively larger effective step.
                */

                v[i][j] = beta2 * v[i][j] + (1.0f - beta2) * g * g;

                /* @doc
                After updating the raw moments, the code removes the early-step
                bias and then updates the parameter:

                $$
                \theta_t = \theta_{t-1} - \alpha
                \dfrac{\hat{m}_t}{\sqrt{\hat{v}_t}+\epsilon}.
                $$

                In code, `m_hat` supplies the direction and `sqrt(v_hat) + eps`
                rescales that direction. The small `eps` value is not a
                learning-rate trick; it is a numerical safeguard for parameters
                whose second moment is near zero.
                */

                float m_hat = m[i][j] / bias_correction1;
                float v_hat = v[i][j] / bias_correction2;
                (*value)[j] -= lr * m_hat / (std::sqrt(v_hat) + eps);
            }
        }
    }

    /* @doc
    ## Learning-Rate Accessors

    The scheduler code talks to optimizers through `get_lr()` and `set_lr()`.
    Keeping these methods here lets `ReduceLROnPlateau` adjust Adam's global
    learning-rate scale without knowing anything about Adam's moment buffers.
    */

    float get_lr() const override { return lr; }

    void set_lr(float new_lr) override { lr = new_lr; }
};
