#include <random>

#include "image.hpp"
#include "vendor/doctest.hpp"

TEST_CASE("Image augmentation preserves shape and labels") {
    ImageDataset dataset{NDArray({2, 28 * 28}), {1, 2}};
    dataset.X.fill(0.0f);
    dataset.X.set({0, 14 * 28 + 14}, 1.0f);
    dataset.X.set({1, 10 * 28 + 10}, 1.0f);

    AugmentConfig config{
        .max_shift = 2,
        .max_rotation_deg = 5.0f,
        .morphology_prob = 1.0f,
        .noise_std = 0.0f,
    };
    std::mt19937 rng(42);
    auto augmented = augment_dataset(dataset, config, rng);

    CHECK(augmented.X.shape == std::vector<int>{2, 28 * 28});
    CHECK(augmented.y == std::vector<int>{1, 2});
    for (int i = 0; i < augmented.X.numel; ++i) {
        CHECK(augmented.X[i] >= 0.0f);
        CHECK(augmented.X[i] <= 1.0f);
    }
}

TEST_CASE("Zero augmentation returns original dataset") {
    ImageDataset dataset{NDArray({1, 28 * 28}), {3}};
    for (int i = 0; i < dataset.X.numel; ++i) {
        dataset.X[i] = static_cast<float>(i % 7) / 6.0f;
    }

    AugmentConfig config{};
    std::mt19937 rng(42);
    auto augmented = augment_dataset(dataset, config, rng);

    CHECK(augmented.y == dataset.y);
    for (int i = 0; i < dataset.X.numel; ++i) {
        CHECK(augmented.X[i] == doctest::Approx(dataset.X[i]));
    }
}

TEST_CASE("Morphology augmentation can expand a binary dot") {
    NDArray image({1, 28 * 28});
    image.fill(0.0f);
    image.set({0, 14 * 28 + 14}, 1.0f);

    AugmentConfig config{
        .max_shift = 0,
        .max_rotation_deg = 0.0f,
        .morphology_prob = 1.0f,
        .noise_std = 0.0f,
    };
    std::mt19937 rng(0);
    auto augmented = augment_bmp_28x28_flat(image, config, rng);

    float sum = 0.0f;
    for (int i = 0; i < augmented.numel; ++i) {
        sum += augmented[i];
        CHECK(augmented[i] >= 0.0f);
        CHECK(augmented[i] <= 1.0f);
    }
    CHECK(sum >= 1.0f);
}
