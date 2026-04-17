#include "module/batchnorm.hpp"

#include <cassert>
#include <cmath>

BatchNorm2D::BatchNorm2D(int channels, float eps, float momentum)
    : channels(channels),
      eps(eps),
      momentum(momentum),
      training(true),
      gamma({1, channels, 1, 1}),
      beta({1, channels, 1, 1}),
      dgamma({1, channels, 1, 1}),
      dbeta({1, channels, 1, 1}),
      running_mean({1, channels, 1, 1}),
      running_var({1, channels, 1, 1}) {
    gamma.fill(1.0f);
    beta.fill(0.0f);
    dgamma.fill(0.0f);
    dbeta.fill(0.0f);
    running_mean.fill(0.0f);
    running_var.fill(1.0f);
}

NDArray BatchNorm2D::forward(const NDArray& x) {
    assert(x.ndim == 4);
    assert(x.shape[1] == channels);

    const int n = x.shape[0];
    const int h = x.shape[2];
    const int w = x.shape[3];
    const float sample_size = static_cast<float>(n * h * w);

    NDArray out(x.shape);
    x_centered_cache = NDArray(x.shape);
    x_hat_cache = NDArray(x.shape);
    inv_std_cache = NDArray({1, channels, 1, 1});

#pragma omp parallel for
    for (int c = 0; c < channels; ++c) {
        float mean = 0.0f;
        float var = 0.0f;

        if (training) {
            for (int i = 0; i < n; ++i) {
                for (int y = 0; y < h; ++y) {
                    for (int xw = 0; xw < w; ++xw) {
                        mean += x.get({i, c, y, xw});
                    }
                }
            }
            mean /= sample_size;

            for (int i = 0; i < n; ++i) {
                for (int y = 0; y < h; ++y) {
                    for (int xw = 0; xw < w; ++xw) {
                        float centered = x.get({i, c, y, xw}) - mean;
                        var += centered * centered;
                    }
                }
            }
            var /= sample_size;

            running_mean.set(
                {0, c, 0, 0},
                (1.0f - momentum) * running_mean.get({0, c, 0, 0}) +
                    momentum * mean);
            running_var.set(
                {0, c, 0, 0},
                (1.0f - momentum) * running_var.get({0, c, 0, 0}) +
                    momentum * var);
        } else {
            mean = running_mean.get({0, c, 0, 0});
            var = running_var.get({0, c, 0, 0});
        }

        float inv_std = 1.0f / std::sqrt(var + eps);
        inv_std_cache.set({0, c, 0, 0}, inv_std);

        for (int i = 0; i < n; ++i) {
            for (int y = 0; y < h; ++y) {
                for (int xw = 0; xw < w; ++xw) {
                    float centered = x.get({i, c, y, xw}) - mean;
                    float x_hat = centered * inv_std;
                    x_centered_cache.set({i, c, y, xw}, centered);
                    x_hat_cache.set({i, c, y, xw}, x_hat);
                    out.set({i, c, y, xw},
                            gamma.get({0, c, 0, 0}) * x_hat +
                                beta.get({0, c, 0, 0}));
                }
            }
        }
    }

    return out;
}

NDArray BatchNorm2D::backward(const NDArray& grad_out) {
    assert(grad_out.ndim == 4);
    assert(grad_out.shape[1] == channels);

    const int n = grad_out.shape[0];
    const int h = grad_out.shape[2];
    const int w = grad_out.shape[3];
    const float sample_size = static_cast<float>(n * h * w);

    NDArray grad_x(grad_out.shape);
    grad_x.fill(0.0f);
    dgamma.fill(0.0f);
    dbeta.fill(0.0f);

#pragma omp parallel for
    for (int c = 0; c < channels; ++c) {
        float grad_gamma = 0.0f;
        float grad_beta = 0.0f;
        float sum_dxhat = 0.0f;
        float sum_dxhat_xhat = 0.0f;

        for (int i = 0; i < n; ++i) {
            for (int y = 0; y < h; ++y) {
                for (int xw = 0; xw < w; ++xw) {
                    float go = grad_out.get({i, c, y, xw});
                    float x_hat = x_hat_cache.get({i, c, y, xw});
                    float dxhat = go * gamma.get({0, c, 0, 0});
                    grad_gamma += go * x_hat;
                    grad_beta += go;
                    sum_dxhat += dxhat;
                    sum_dxhat_xhat += dxhat * x_hat;
                }
            }
        }

        dgamma.set({0, c, 0, 0}, grad_gamma);
        dbeta.set({0, c, 0, 0}, grad_beta);

        float inv_std = inv_std_cache.get({0, c, 0, 0});
        for (int i = 0; i < n; ++i) {
            for (int y = 0; y < h; ++y) {
                for (int xw = 0; xw < w; ++xw) {
                    float go = grad_out.get({i, c, y, xw});
                    float x_hat = x_hat_cache.get({i, c, y, xw});
                    float dxhat = go * gamma.get({0, c, 0, 0});
                    float value = (sample_size * dxhat - sum_dxhat -
                                   x_hat * sum_dxhat_xhat) *
                                  (inv_std / sample_size);
                    grad_x.set({i, c, y, xw}, value);
                }
            }
        }
    }

    return grad_x;
}

void BatchNorm2D::step(float lr) {
    gamma = gamma - dgamma * lr;
    beta = beta - dbeta * lr;
}

void BatchNorm2D::train() { training = true; }

void BatchNorm2D::eval() { training = false; }

std::vector<Parameter> BatchNorm2D::parameters() {
    return {
        {&gamma, &dgamma},
        {&beta, &dbeta},
        {&running_mean, nullptr},
        {&running_var, nullptr},
    };
}
