#include <filesystem>
#include <string>

#include "checkpoint.hpp"
#include "module/batchnorm.hpp"
#include "module/dense.hpp"
#include "vendor/doctest.hpp"

using namespace std;
namespace fs = std::filesystem;

TEST_CASE("Checkpoint save and load restores linear parameters") {
    Linear original(3, 2);
    Linear restored(3, 2);
    const string config_text =
        "[model]\n"
        "hidden_size = 32\n"
        "[train]\n"
        "lr = 0.001\n";

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
                    "eval_loss", config_text);

    int epoch = 0;
    float metric_value = 0.0f;
    string metric_name;
    string loaded_config_text;
    CHECK(load_checkpoint(checkpoint_path.string(), restored.parameters(), &epoch,
                          &metric_value, &metric_name, &loaded_config_text));

    auto y_original = original.forward(x);
    auto y_restored = restored.forward(x);
    for (int i = 0; i < y_original.numel; ++i) {
        CHECK(y_original[i] == doctest::Approx(y_restored[i]));
    }
    CHECK(epoch == 7);
    CHECK(metric_value == doctest::Approx(0.123f));
    CHECK(metric_name == "eval_loss");
    CHECK(loaded_config_text == config_text);

    fs::remove(checkpoint_path);
}

TEST_CASE("Checkpoint load fails on shape mismatch") {
    Linear original(3, 2);
    Linear mismatched(4, 2);
    fs::path checkpoint_path =
        fs::temp_directory_path() / "lab1_checkpoint_mismatch.ckpt";

    save_checkpoint(checkpoint_path.string(), original.parameters(), 1, 1.0f,
                    "eval_loss", "[model]\nwidth = 3\n");
    CHECK_FALSE(load_checkpoint(checkpoint_path.string(),
                                mismatched.parameters()));

    fs::remove(checkpoint_path);
}

TEST_CASE("Checkpoint metadata can be loaded without model parameters") {
    Linear original(3, 2);
    fs::path checkpoint_path =
        fs::temp_directory_path() / "lab1_checkpoint_metadata.ckpt";
    const string config_text = "[data]\neval_dir = data/test\n";

    save_checkpoint(checkpoint_path.string(), original.parameters(), 5, 0.95f,
                    "eval_acc", config_text);

    CheckpointMetadata metadata;
    CHECK(load_checkpoint_metadata(checkpoint_path.string(), &metadata));
    CHECK(metadata.epoch == 5);
    CHECK(metadata.metric_value == doctest::Approx(0.95f));
    CHECK(metadata.metric_name == "eval_acc");
    CHECK(metadata.config_text == config_text);

    fs::remove(checkpoint_path);
}

TEST_CASE("Checkpoint save and load restores batchnorm running stats") {
    BatchNorm2D original(2);
    BatchNorm2D restored(2);
    fs::path checkpoint_path =
        fs::temp_directory_path() / "lab1_checkpoint_batchnorm.ckpt";

    NDArray x({2, 2, 2, 2});
    for (int i = 0; i < x.numel; ++i) x[i] = static_cast<float>(i + 1);

    original.train();
    original.forward(x);
    original.eval();

    save_checkpoint(checkpoint_path.string(), original.parameters(), 3, 0.25f,
                    "eval_loss", "[model]\nkind = batchnorm\n");
    CHECK(load_checkpoint(checkpoint_path.string(), restored.parameters()));

    restored.eval();
    auto y_original = original.forward(x);
    auto y_restored = restored.forward(x);
    for (int i = 0; i < y_original.numel; ++i) {
        CHECK(y_original[i] == doctest::Approx(y_restored[i]).epsilon(1e-6));
    }

    fs::remove(checkpoint_path);
}
