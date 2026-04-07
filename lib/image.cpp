#include "image.hpp"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "vendor/stb_image.hpp"

using namespace std;
namespace fs = std::filesystem;

static float normalize_pixel(unsigned char pixel, bool invert) {
    float value = static_cast<float>(pixel) / 255.0f;
    return invert ? 1.0f - value : value;
}

static bool has_bmp_extension(const fs::path& path) {
    auto ext = path.extension().string();
    std::transform(ext.begin(), ext.end(), ext.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return ext == ".bmp";
}

NDArray load_bmp_28x28_flat(const string& path, bool invert) {
    int width = 0;
    int height = 0;
    int channels = 0;
    unsigned char* data =
        stbi_load(path.c_str(), &width, &height, &channels, 1);
    if (data == nullptr) {
        throw runtime_error("failed to load image: " + path);
    }

    if (width != 28 || height != 28) {
        stbi_image_free(data);
        throw runtime_error("expected a 28x28 BMP image: " + path);
    }

    NDArray image({1, 28 * 28});
    for (int i = 0; i < 28 * 28; ++i) {
        image[i] = normalize_pixel(data[i], invert);
    }

    stbi_image_free(data);
    return image;
}

ImageDataset load_bmp_dataset(const string& root_dir, bool invert) {
    vector<pair<string, int>> samples;

    for (int label = 1; label <= 12; ++label) {
        fs::path class_dir = fs::path(root_dir) / to_string(label);
        if (!fs::exists(class_dir) || !fs::is_directory(class_dir)) {
            continue;
        }

        for (const auto& entry : fs::directory_iterator(class_dir)) {
            if (!entry.is_regular_file() || !has_bmp_extension(entry.path())) {
                continue;
            }
            samples.emplace_back(entry.path().string(), label);
        }
    }

    sort(samples.begin(), samples.end(), [](const auto& lhs, const auto& rhs) {
        return lhs.first < rhs.first;
    });

    ImageDataset dataset{NDArray({static_cast<int>(samples.size()), 28 * 28}),
                         {}};
    dataset.y.reserve(samples.size());

    for (int i = 0; i < static_cast<int>(samples.size()); ++i) {
        NDArray image = load_bmp_28x28_flat(samples[i].first, invert);
        for (int j = 0; j < 28 * 28; ++j) {
            dataset.X.set({i, j}, image.get({0, j}));
        }
        dataset.y.push_back(samples[i].second);
    }

    return dataset;
}

NDArray augment_bmp_28x28_flat(const NDArray& image, const AugmentConfig& config,
                               mt19937& rng) {
    if (image.shape != vector<int>{1, 28 * 28}) {
        throw runtime_error("augment_bmp_28x28_flat expects shape [1, 784]");
    }

    NDArray augmented({1, 28 * 28});
    augmented.fill(0.0f);

    uniform_int_distribution<int> shift_dist(-config.max_shift, config.max_shift);
    int shift_x = config.max_shift > 0 ? shift_dist(rng) : 0;
    int shift_y = config.max_shift > 0 ? shift_dist(rng) : 0;

    for (int y = 0; y < 28; ++y) {
        for (int x = 0; x < 28; ++x) {
            int src_x = x - shift_x;
            int src_y = y - shift_y;
            float value = 0.0f;
            if (src_x >= 0 && src_x < 28 && src_y >= 0 && src_y < 28) {
                value = image.get({0, src_y * 28 + src_x});
            }
            augmented.set({0, y * 28 + x}, value);
        }
    }

    if (config.noise_std > 0.0f) {
        normal_distribution<float> noise_dist(0.0f, config.noise_std);
        for (int i = 0; i < augmented.numel; ++i) {
            float value = augmented[i] + noise_dist(rng);
            augmented[i] = clamp(value, 0.0f, 1.0f);
        }
    }

    return augmented;
}

ImageDataset augment_dataset(const ImageDataset& dataset,
                             const AugmentConfig& config, mt19937& rng) {
    if (config.max_shift == 0 && config.noise_std == 0.0f) {
        return dataset;
    }

    ImageDataset augmented{
        NDArray({dataset.X.shape[0], dataset.X.shape[1]}),
        dataset.y,
    };

    for (int i = 0; i < dataset.X.shape[0]; ++i) {
        NDArray row = dataset.X.slice(i, i + 1);
        NDArray aug_row = augment_bmp_28x28_flat(row, config, rng);
        for (int j = 0; j < dataset.X.shape[1]; ++j) {
            augmented.X.set({i, j}, aug_row.get({0, j}));
        }
    }

    return augmented;
}
