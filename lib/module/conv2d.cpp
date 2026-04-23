#include "module/conv2d.hpp"

#include <cassert>

#include "ndarray.hpp"
/* @doc
# Conv2D

This file implements a 2D convolution layer for tensors in `NCHW` layout:
batch, channel, height, width. The layer stores one square kernel per
`(out_channel, in_channel)` pair, so the learnable weight tensor has shape
`(C_out, C_in, K, K)`, and the bias tensor has shape `(1, C_out, 1, 1)`.

For an input $X \in \mathbb{R}^{N \times C_{in} \times H \times W}$, the
output spatial dimensions are

$$
H_{out} = \left\lfloor \frac{H + 2P - K}{S} \right\rfloor + 1,
\qquad
W_{out} = \left\lfloor \frac{W + 2P - K}{S} \right\rfloor + 1.
$$

The direct definition of convolution is a nested sum:

$$
Y_{n,o,h,w} =
b_o +
\sum_{i=0}^{C_{in}-1}
\sum_{r=0}^{K-1}
\sum_{c=0}^{K-1}
X_{n,i,hS+r-P,wS+c-P} W_{o,i,r,c}.
$$

This implementation does not evaluate that equation as one monolithic set of
loops. Instead, it lowers the input into a matrix with `im2col`, reshapes the
weights into another matrix, uses matrix multiplication for the expensive
part, and reshapes the result back to `NCHW`.
*/

NDArray im2col(const NDArray& x, int kernel_size, int stride, int padding) {
    const int n = x.shape[0];
    const int channels = x.shape[1];
    const int height = x.shape[2];
    const int width = x.shape[3];
    const int h_out = (height + 2 * padding - kernel_size) / stride + 1;
    const int w_out = (width + 2 * padding - kernel_size) / stride + 1;

    NDArray cols({n * h_out * w_out, channels * kernel_size * kernel_size});
    cols.fill(0.0f);

/* @doc
## IM2Col

`im2col` turns every receptive field of the input into one row of a matrix.
The matrix has

- one row for every output location in every batch item: $N H_{out} W_{out}$ rows,
- one column for every value touched by a kernel:
  $C_{in} K^2$ columns.

A row index identifies the output coordinate `(batch, oh, ow)`:

$$
row = (batch \cdot H_{out} + oh) \cdot W_{out} + ow.
$$

A column index identifies one input-channel and kernel-offset pair:

$$
col = (channel \cdot K + kh) \cdot K + kw.
$$

The corresponding input coordinate is

$$
ih = oh \cdot S + kh - P,\qquad iw = ow \cdot S + kw - P.
$$

If that coordinate falls outside the input image, the lowered matrix stores
zero. That is exactly how zero-padding appears in the matrix form: padding is
not materialized as a larger image, it is generated while each receptive field
is copied.
*/

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

/* @doc
## Col2Im

`col2im` is the reverse accumulation used during backpropagation. The
gradient with respect to the lowered matrix has the same shape as the matrix
produced by `im2col`: `(N H_out W_out, C_in K^2)`.

Reversing `im2col` is not just a copy. Neighboring receptive fields overlap,
so a single input pixel may contribute to many rows of the lowered matrix.
During the backward pass, all of those contributions must be summed into the
same input-gradient location:

$$
\frac{\partial L}{\partial X_{n,c,h,w}}
=
\sum_{\text{windows covering }(h,w)}
\frac{\partial L}{\partial cols_{row,col}}.
$$

This is why the assignment below reads the old value from `x`, adds the
contribution from `cols`, and writes the sum back.
*/

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

/* @doc
## Weight Matrix View

Convolution weights are stored in the natural 4D shape
`(C_out, C_in, K, K)`, but the lowered input is a 2D matrix. To use matrix
multiplication, the weights are viewed as

$$
W_{2d} \in \mathbb{R}^{(C_{in} K^2) \times C_{out}}.
$$

Each row corresponds to the same `(input channel, kernel row, kernel column)`
coordinate used by `im2col`. Each column corresponds to one output channel.
With that layout, the convolution core becomes

$$
out_{2d} = cols \cdot W_{2d},
$$

producing a matrix with shape `(N H_out W_out, C_out)`.
*/

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

/* @doc
## Restoring Weight Gradients

The backward pass computes the weight gradient in the same 2D layout used for
matrix multiplication. `weights_from_2d` maps that gradient back to the
layer's 4D tensor layout so the optimizer can update `W` without knowing
anything about the im2col representation.

This pair of functions is deliberately symmetric: `weights_to_2d` defines the
coordinate transform, and `weights_from_2d` applies the inverse transform.
*/

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

/* @doc
## Layer State

A `Conv2D` object owns both parameters and gradient buffers:

- `W` stores the convolution kernels,
- `b` stores one scalar bias per output channel,
- `dW` and `db` store gradients with matching shapes.

The constructor records the structural hyperparameters (`in_channels`,
`out_channels`, `kernel_size`, `stride`, and `padding`) and initializes the
parameters. Weights use the project-wide `init_weights` helper, while biases
start at zero.
*/

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

/* @doc
## Forward Pass

The forward pass checks the input contract, caches the input for the backward
pass, computes the output size, and lowers the input with `im2col`.

After lowering, the expensive part is a single matrix multiplication:

$$
cols \in \mathbb{R}^{(N H_{out} W_{out}) \times (C_{in}K^2)},\quad
W_{2d} \in \mathbb{R}^{(C_{in}K^2) \times C_{out}},
$$

$$
out_{2d} = cols W_{2d}.
$$

The final loop reshapes `out2d` back into `(N, C_out, H_out, W_out)` and adds
the channel bias. The bias is broadcast over batch and spatial dimensions:

$$
Y_{n,o,h,w} = out2d_{row,o} + b_o.
$$
*/

NDArray Conv2D::forward(const NDArray& x) {
    assert(x.ndim == 4);
    assert(x.shape[1] == in_channels);

    input_cache = x;
    const int n = x.shape[0];
    const int height = x.shape[2];
    const int width = x.shape[3];
    const int h_out = (height + 2 * padding - kernel_size) / stride + 1;
    const int w_out = (width + 2 * padding - kernel_size) / stride + 1;

/* @doc
`col_cache` is saved because the weight gradient is easiest to compute from
the lowered input:

$$
\frac{\partial L}{\partial W_{2d}} = cols^T
\frac{\partial L}{\partial out_{2d}}.
$$

Keeping the cache avoids recomputing `im2col` during `backward`.
*/

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

/* @doc
## Backward Pass

The incoming gradient `grad_out` has the same layout as the forward output:
`(N, C_out, H_out, W_out)`. The backward pass reshapes it into
`grad2d`, aligned with the rows of `out2d`.

From the matrix equation

$$
out_{2d} = cols W_{2d},
$$

the gradients follow the standard matrix-calculus identities:

$$
\frac{\partial L}{\partial W_{2d}} = cols^T
\frac{\partial L}{\partial out_{2d}},
\qquad
\frac{\partial L}{\partial cols} =
\frac{\partial L}{\partial out_{2d}} W_{2d}^T.
$$

The bias gradient is the sum of `grad2d` over every batch and spatial row:

$$
\frac{\partial L}{\partial b_o}
=
\sum_{row}
\frac{\partial L}{\partial out_{2d,row,o}}.
$$

Finally, `col2im` accumulates the lowered input gradient back into the
original input shape.
*/

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
