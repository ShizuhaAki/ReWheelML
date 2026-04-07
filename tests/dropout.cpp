#include "module/dropout.hpp"

#include "ndarray.hpp"
#include "vendor/doctest.hpp"

TEST_CASE("Dropout with zero probability is identity") {
    Dropout dropout(0.0f, 42);
    NDArray x({2, 3});
    for (int i = 0; i < x.numel; ++i) x[i] = static_cast<float>(i + 1);

    auto y = dropout.forward(x);
    for (int i = 0; i < x.numel; ++i) {
        CHECK(y[i] == doctest::Approx(x[i]));
    }

    auto grad = dropout.backward(x);
    for (int i = 0; i < x.numel; ++i) {
        CHECK(grad[i] == doctest::Approx(x[i]));
    }
}

TEST_CASE("Dropout eval mode is identity") {
    Dropout dropout(0.5f, 42);
    dropout.eval();

    NDArray x({2, 3});
    for (int i = 0; i < x.numel; ++i) x[i] = static_cast<float>(i + 1);

    auto y = dropout.forward(x);
    for (int i = 0; i < x.numel; ++i) {
        CHECK(y[i] == doctest::Approx(x[i]));
    }
}

TEST_CASE("Dropout backward uses forward mask") {
    Dropout dropout(0.5f, 42);
    NDArray x({2, 4});
    x.fill(1.0f);

    auto y = dropout.forward(x);
    NDArray grad_out({2, 4});
    grad_out.fill(1.0f);
    auto grad = dropout.backward(grad_out);

    for (int i = 0; i < y.numel; ++i) {
        CHECK(grad[i] == doctest::Approx(y[i]));
    }
}
