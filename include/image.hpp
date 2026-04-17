#pragma once

#include <string>
#include <random>
#include <vector>

#include "ndarray.hpp"

struct ImageDataset {
    NDArray X;
    std::vector<int> y;
};

struct AugmentConfig {
    int max_shift = 0;
    float max_rotation_deg = 0.0f;
    float morphology_prob = 0.0f;
    float noise_std = 0.0f;
};

NDArray load_bmp_28x28_flat(const std::string& path, bool invert = false);
ImageDataset load_bmp_dataset(const std::string& root_dir, bool invert = false);
NDArray augment_bmp_28x28_flat(const NDArray& image, const AugmentConfig& config,
                               std::mt19937& rng);
ImageDataset augment_dataset(const ImageDataset& dataset,
                             const AugmentConfig& config, std::mt19937& rng);
