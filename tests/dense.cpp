#include "module/dense.hpp"

#include "functional.hpp"
#include "ndarray.hpp"
#include "optim/optim.hpp"
#include "vendor/doctest.hpp"

TEST_CASE("Dense layer forward, backward, and step") {
    int input_dim = 3;
    int output_dim = 2;
    Linear linear(input_dim, output_dim);
    Activation activation("sigmoid");

    // Input batch of 2 samples
    NDArray x({2, input_dim});
    x.fill(1.0);

    // Forward pass
    NDArray y1 = activation.forward(linear.forward(x));
    CHECK(y1.shape == std::vector<int>{2, output_dim});
    for (int i = 0; i < y1.numel; ++i) {
        CHECK(y1[i] >= 0.0);
        CHECK(y1[i] <= 1.0);
    }

    // Backward pass
    NDArray dY({2, output_dim});
    dY.fill(1.0);
    NDArray dX = linear.backward(activation.backward(dY));
    CHECK(dX.shape == std::vector<int>{2, input_dim});

    // Forward pass again before step
    NDArray y_before_step = activation.forward(linear.forward(x));

    // Apply optimizer step
    SGD optimizer(linear.parameters(), 0.1f);
    optimizer.step();
    optimizer.zero_grad();

    // Forward pass after step
    NDArray y_after_step = activation.forward(linear.forward(x));

    // The output should change after updating parameters
    bool output_changed = false;
    for (int i = 0; i < y_before_step.numel; ++i) {
        if (y_before_step[i] != y_after_step[i]) {
            output_changed = true;
            break;
        }
    }
    CHECK(output_changed);
}
