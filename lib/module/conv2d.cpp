#include "module/conv2d.hpp"

#include <cassert>

#include "ndarray.hpp"

NDArray im2col(const NDArray& x, int kernel_size, int stride, int padding) {
    const int n = x.shape[0];
    const int channels = x.shape[1];
    const int height = x.shape[2];
    const int width = x.shape[3];
    const int h_out = (height + 2 * padding - kernel_size) / stride + 1;
    const int w_out = (width + 2 * padding - kernel_size) / stride + 1;

    NDArray cols({n * h_out * w_out, channels * kernel_size * kernel_size});
    cols.fill(0.0f);

#pragma omp parallel for
    for (int batch = 0; batch < n; ++batch) {
        for (int oh = 0; oh < h_out; ++oh) {
            for (int ow = 0; ow < w_out; ++ow) {
                const int row = (batch * h_out + oh) * w_out + ow;
                for (int channel = 0; channel < channels; ++channel) {
                    for (int kh = 0; kh < kernel_size; ++kh) {
                        for (int kw = 0; kw < kernel_size; ++kw) {
                            const int ih = oh * stride + kh - padding;
                            const int iw = ow * stride + kw - padding;
                            const int col =
                                ((channel * kernel_size) + kh) * kernel_size +
                                kw;
                            if (ih < 0 || ih >= height || iw < 0 ||
                                iw >= width) {
                                cols.set({row, col}, 0.0f);
                                continue;
                            }
                            cols.set({row, col},
                                     x.get({batch, channel, ih, iw}));
                        }
                    }
                }
            }
        }
    }

    return cols;
}

NDArray col2im(const NDArray& cols, int n, int channels, int height, int width,
               int kernel_size, int stride, int padding) {
    const int h_out = (height + 2 * padding - kernel_size) / stride + 1;
    const int w_out = (width + 2 * padding - kernel_size) / stride + 1;

    NDArray x({n, channels, height, width});
    x.fill(0.0f);

#pragma omp parallel for
    for (int batch = 0; batch < n; ++batch) {
        for (int oh = 0; oh < h_out; ++oh) {
            for (int ow = 0; ow < w_out; ++ow) {
                const int row = (batch * h_out + oh) * w_out + ow;
                for (int channel = 0; channel < channels; ++channel) {
                    for (int kh = 0; kh < kernel_size; ++kh) {
                        for (int kw = 0; kw < kernel_size; ++kw) {
                            const int ih = oh * stride + kh - padding;
                            const int iw = ow * stride + kw - padding;
                            if (ih < 0 || ih >= height || iw < 0 ||
                                iw >= width) {
                                continue;
                            }
                            const int col =
                                ((channel * kernel_size) + kh) * kernel_size +
                                kw;
                            x.set({batch, channel, ih, iw},
                                  x.get({batch, channel, ih, iw}) +
                                      cols.get({row, col}));
                        }
                    }
                }
            }
        }
    }

    return x;
}

NDArray weights_to_2d(const NDArray& w) {
    const int out_channels = w.shape[0];
    const int in_channels = w.shape[1];
    const int kernel_size = w.shape[2];
    NDArray w2d({in_channels * kernel_size * kernel_size, out_channels});

#pragma omp parallel for collapse(2)
    for (int oc = 0; oc < out_channels; ++oc) {
        for (int ic = 0; ic < in_channels; ++ic) {
            for (int kh = 0; kh < kernel_size; ++kh) {
                for (int kw = 0; kw < kernel_size; ++kw) {
                    const int row =
                        ((ic * kernel_size) + kh) * kernel_size + kw;
                    w2d.set({row, oc}, w.get({oc, ic, kh, kw}));
                }
            }
        }
    }

    return w2d;
}

void weights_from_2d(const NDArray& w2d, NDArray& w) {
    const int out_channels = w.shape[0];
    const int in_channels = w.shape[1];
    const int kernel_size = w.shape[2];

#pragma omp parallel for collapse(2)
    for (int oc = 0; oc < out_channels; ++oc) {
        for (int ic = 0; ic < in_channels; ++ic) {
            for (int kh = 0; kh < kernel_size; ++kh) {
                for (int kw = 0; kw < kernel_size; ++kw) {
                    const int row =
                        ((ic * kernel_size) + kh) * kernel_size + kw;
                    w.set({oc, ic, kh, kw}, w2d.get({row, oc}));
                }
            }
        }
    }
}

Conv2D::Conv2D(int in_channels, int out_channels, int kernel_size, int stride,
               int padding)
    : in_channels(in_channels),
      out_channels(out_channels),
      kernel_size(kernel_size),
      stride(stride),
      padding(padding),
      W({out_channels, in_channels, kernel_size, kernel_size}),
      b({1, out_channels, 1, 1}),
      dW({out_channels, in_channels, kernel_size, kernel_size}),
      db({1, out_channels, 1, 1}) {
    W.init_weights();
    b.fill(0.0f);
}

NDArray Conv2D::forward(const NDArray& x) {
    assert(x.ndim == 4);
    assert(x.shape[1] == in_channels);

    input_cache = x;
    const int n = x.shape[0];
    const int height = x.shape[2];
    const int width = x.shape[3];
    const int h_out = (height + 2 * padding - kernel_size) / stride + 1;
    const int w_out = (width + 2 * padding - kernel_size) / stride + 1;

    col_cache = im2col(x, kernel_size, stride, padding);
    NDArray w2d = weights_to_2d(W);
    NDArray out2d = col_cache.matmul(w2d);

    NDArray out({n, out_channels, h_out, w_out});
#pragma omp parallel for
    for (int batch = 0; batch < n; ++batch) {
        for (int oh = 0; oh < h_out; ++oh) {
            for (int ow = 0; ow < w_out; ++ow) {
                const int row = (batch * h_out + oh) * w_out + ow;
                for (int oc = 0; oc < out_channels; ++oc) {
                    out.set({batch, oc, oh, ow},
                            out2d.get({row, oc}) + b.get({0, oc, 0, 0}));
                }
            }
        }
    }

    return out;
}

NDArray Conv2D::backward(const NDArray& grad_out) {
    assert(grad_out.ndim == 4);
    assert(grad_out.shape[1] == out_channels);

    const int n = input_cache.shape[0];
    const int height = input_cache.shape[2];
    const int width = input_cache.shape[3];
    const int h_out = grad_out.shape[2];
    const int w_out = grad_out.shape[3];

    NDArray grad2d({n * h_out * w_out, out_channels});
#pragma omp parallel for
    for (int batch = 0; batch < n; ++batch) {
        for (int oh = 0; oh < h_out; ++oh) {
            for (int ow = 0; ow < w_out; ++ow) {
                const int row = (batch * h_out + oh) * w_out + ow;
                for (int oc = 0; oc < out_channels; ++oc) {
                    grad2d.set({row, oc}, grad_out.get({batch, oc, oh, ow}));
                }
            }
        }
    }

    db.fill(0.0f);
    for (int row = 0; row < grad2d.shape[0]; ++row) {
        for (int oc = 0; oc < out_channels; ++oc) {
            db.set({0, oc, 0, 0},
                   db.get({0, oc, 0, 0}) + grad2d.get({row, oc}));
        }
    }

    NDArray dW2d = col_cache.transpose2d().matmul(grad2d);
    weights_from_2d(dW2d, dW);

    NDArray w2d = weights_to_2d(W);
    NDArray dcols = grad2d.matmul(w2d.transpose2d());
    NDArray dX = col2im(dcols, n, in_channels, height, width, kernel_size,
                        stride, padding);
    return dX;
}

void Conv2D::step(float lr) {
    W = W - dW * lr;
    b = b - db * lr;
}

std::vector<Parameter> Conv2D::parameters() { return {{&W, &dW}, {&b, &db}}; }
