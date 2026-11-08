#include "module/pooling.hpp"

#include <cassert>
#include <cfloat>

#include "ndarray.hpp"

/* @doc
# MaxPool2D

This file implements max pooling for 4D feature maps in `NCHW` layout. Pooling
does not mix channels: each output value is computed from one square spatial
window inside the same `(batch, channel)` slice.

For input $X \in \mathbb{R}^{N \times C \times H \times W}$, pool size $K$,
and stride $S$, the output spatial dimensions are

$$
H_{out} = \left\lfloor \frac{H - K}{S} \right\rfloor + 1,
\qquad
W_{out} = \left\lfloor \frac{W - K}{S} \right\rfloor + 1.
$$

There is no padding in this implementation. Windows that would extend past the
right or bottom edge are not emitted.
*/

MaxPool2D::MaxPool2D(int pool_size, int stride)
    : pool_size(pool_size), stride(stride) {}

/* @doc
## Forward Pass

For every output coordinate `(n, c, oh, ow)`, the corresponding input window
starts at

$$
ih_0 = oh \cdot S,\qquad iw_0 = ow \cdot S.
$$

The output is the largest value in the $K \times K$ window:

$$
Y_{n,c,oh,ow}
=
\max_{0 \le ph,pw < K}
X_{n,c,ohS+ph,owS+pw}.
$$

Besides writing the maximum value, the forward pass records the input location
where that maximum occurred. `max_h_cache` stores the winning input row and
`max_w_cache` stores the winning input column for each output element. Those
argmax caches are the only information needed to route gradients in the
backward pass.
*/

NDArray MaxPool2D::forward(const NDArray& x) {
    assert(x.ndim == 4);

    input_cache = x;
    int N = x.shape[0], C = x.shape[1], H = x.shape[2], W = x.shape[3];
    int H_out = (H - pool_size) / stride + 1;
    int W_out = (W - pool_size) / stride + 1;

    NDArray out({N, C, H_out, W_out});
    out.fill(0.0f);

    max_h_cache = NDArray({N, C, H_out, W_out});
    max_w_cache = NDArray({N, C, H_out, W_out});

#pragma omp parallel for collapse(2)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int oh = 0; oh < H_out; oh++) {
                for (int ow = 0; ow < W_out; ow++) {
                    float best = -FLT_MAX;
                    int best_h = -1;
                    int best_w = -1;
                    for (int ph = 0; ph < pool_size; ph++) {
                        for (int pw = 0; pw < pool_size; pw++) {
                            int ih = oh * stride + ph;
                            int iw = ow * stride + pw;
                            float v = x.get({n, c, ih, iw});
                            if (v > best) {
                                best = v;
                                best_h = ih;
                                best_w = iw;
                            }
                        }
                    }
                    out.set({n, c, oh, ow}, best);
                    max_h_cache.set({n, c, oh, ow}, static_cast<float>(best_h));
                    max_w_cache.set({n, c, oh, ow}, static_cast<float>(best_w));
                }
            }
        }
    }
    return out;
}

/* @doc
## Backward Pass

Max pooling has no gradient for non-maximum elements in a window. During
backpropagation, each upstream gradient is added only to the input element that
won the corresponding forward window:

$$
\frac{\partial L}{\partial X_{n,c,ih,iw}}
\mathrel{+}=
\frac{\partial L}{\partial Y_{n,c,oh,ow}}
\quad \text{where } (ih, iw) = argmax(oh, ow).
$$

The implementation reads `(ih, iw)` from the argmax caches populated by
`forward`. The input-gradient tensor starts at zero, so if overlapping pooling
windows choose the same input element, their gradient contributions are summed.
*/

NDArray MaxPool2D::backward(const NDArray& grad_out) {
    assert(grad_out.ndim == 4);
    
    int N = input_cache.shape[0];
    int C = input_cache.shape[1];
    int H = input_cache.shape[2];
    int W = input_cache.shape[3];
    int H_out = grad_out.shape[2];
    int W_out = grad_out.shape[3];

    NDArray dX({N, C, H, W});
    dX.fill(0.0f);

#pragma omp parallel for collapse(2)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int oh = 0; oh < H_out; oh++) {
                for (int ow = 0; ow < W_out; ow++) {
                    float go = grad_out.get({n, c, oh, ow});

                    int ih = static_cast<int>(max_h_cache.get({n, c, oh, ow}));
                    int iw = static_cast<int>(max_w_cache.get({n, c, oh, ow}));

                    dX.set({n, c, ih, iw}, dX.get({n, c, ih, iw}) + go);
                }
            }
        }
    }

    return dX;
}
