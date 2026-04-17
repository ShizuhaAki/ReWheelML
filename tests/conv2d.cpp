#include "module/conv2d.hpp"

#include <vector>

#include "vendor/doctest.hpp"

namespace {
NDArray naive_conv_forward(const NDArray& x, const NDArray& w, const NDArray& b,
                           int stride, int padding) {
    const int n = x.shape[0];
    const int in_channels = x.shape[1];
    const int height = x.shape[2];
    const int width = x.shape[3];
    const int out_channels = w.shape[0];
    const int kernel_size = w.shape[2];
    const int h_out = (height + 2 * padding - kernel_size) / stride + 1;
    const int w_out = (width + 2 * padding - kernel_size) / stride + 1;

    NDArray out({n, out_channels, h_out, w_out});
    out.fill(0.0f);

    for (int batch = 0; batch < n; ++batch) {
        for (int oc = 0; oc < out_channels; ++oc) {
            for (int oh = 0; oh < h_out; ++oh) {
                for (int ow = 0; ow < w_out; ++ow) {
                    float sum = b.get({0, oc, 0, 0});
                    for (int ic = 0; ic < in_channels; ++ic) {
                        for (int kh = 0; kh < kernel_size; ++kh) {
                            for (int kw = 0; kw < kernel_size; ++kw) {
                                int ih = oh * stride + kh - padding;
                                int iw = ow * stride + kw - padding;
                                if (ih < 0 || ih >= height || iw < 0 ||
                                    iw >= width) {
                                    continue;
                                }
                                sum += x.get({batch, ic, ih, iw}) *
                                       w.get({oc, ic, kh, kw});
                            }
                        }
                    }
                    out.set({batch, oc, oh, ow}, sum);
                }
            }
        }
    }

    return out;
}

NDArray make_input(const std::vector<int>& shape) {
    NDArray x(shape);
    for (int i = 0; i < x.numel; ++i) {
        x[i] = static_cast<float>((i % 11) - 5) * 0.25f;
    }
    return x;
}
}  // namespace

TEST_CASE("Conv2D forward matches naive multi-channel convolution") {
    Conv2D conv(2, 3, 3, 1, 1);
    auto params = conv.parameters();

    NDArray x = make_input({2, 2, 5, 5});
    NDArray expected = naive_conv_forward(x, *params[0].value, *params[1].value,
                                          1, 1);
    NDArray actual = conv.forward(x);

    CHECK(actual.shape == expected.shape);
    for (int i = 0; i < actual.numel; ++i) {
        CHECK(actual[i] == doctest::Approx(expected[i]).epsilon(1e-5));
    }
}

TEST_CASE("Conv2D backward returns correct shapes for strided padded conv") {
    Conv2D conv(2, 3, 3, 2, 1);
    NDArray x = make_input({2, 2, 6, 6});
    NDArray out = conv.forward(x);

    NDArray grad_out(out.shape);
    grad_out.fill(1.0f);
    NDArray grad_x = conv.backward(grad_out);
    auto params = conv.parameters();

    CHECK(out.shape == std::vector<int>{2, 3, 3, 3});
    CHECK(grad_x.shape == x.shape);
    CHECK(params[0].grad->shape == std::vector<int>{3, 2, 3, 3});
    CHECK(params[1].grad->shape == std::vector<int>{1, 3, 1, 1});
}
