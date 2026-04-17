#include "module/pooling.hpp"

#include <cassert>
#include <cfloat>

#include "ndarray.hpp"

MaxPool2D::MaxPool2D(int pool_size, int stride)
    : pool_size(pool_size), stride(stride) {}

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
