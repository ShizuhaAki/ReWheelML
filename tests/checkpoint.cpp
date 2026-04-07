#include <filesystem>
#include <string>

#include "checkpoint.hpp"
#include "module/dense.hpp"
#include "vendor/doctest.hpp"

using namespace std;
namespace fs = std::filesystem;

TEST_CASE("Checkpoint save and load restores linear parameters") {
    Linear original(3, 2);
    Linear restored(3, 2);

    NDArray x({2, 3});
    for (int i = 0; i < x.numel; ++i) x[i] = static_cast<float>(i + 1);

    NDArray grad_out({2, 2});
    grad_out.fill(1.0f);

    original.forward(x);
    original.backward(grad_out);
    original.step(0.05f);

    fs::path checkpoint_path =
        fs::temp_directory_path() / "lab1_checkpoint_roundtrip.ckpt";
    save_checkpoint(checkpoint_path.string(), original.parameters(), 7, 0.123f,
                    "eval_loss");

    int epoch = 0;
    float metric_value = 0.0f;
    string metric_name;
    CHECK(load_checkpoint(checkpoint_path.string(), restored.parameters(), &epoch,
                          &metric_value, &metric_name));

    auto y_original = original.forward(x);
    auto y_restored = restored.forward(x);
    for (int i = 0; i < y_original.numel; ++i) {
        CHECK(y_original[i] == doctest::Approx(y_restored[i]));
    }
    CHECK(epoch == 7);
    CHECK(metric_value == doctest::Approx(0.123f));
    CHECK(metric_name == "eval_loss");

    fs::remove(checkpoint_path);
}

TEST_CASE("Checkpoint load fails on shape mismatch") {
    Linear original(3, 2);
    Linear mismatched(4, 2);
    fs::path checkpoint_path =
        fs::temp_directory_path() / "lab1_checkpoint_mismatch.ckpt";

    save_checkpoint(checkpoint_path.string(), original.parameters(), 1, 1.0f,
                    "eval_loss");
    CHECK_FALSE(load_checkpoint(checkpoint_path.string(),
                                mismatched.parameters()));

    fs::remove(checkpoint_path);
}
