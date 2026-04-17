#pragma once

#include <string>
#include <vector>

#include "module/module.hpp"

struct CheckpointMetadata {
    int epoch = 0;
    float metric_value = 0.0f;
    std::string metric_name;
    std::string config_text;
};

void save_checkpoint(const std::string& path,
                     const std::vector<Parameter>& params, int epoch,
                     float metric_value, const std::string& metric_name,
                     const std::string& config_text = "");

bool load_checkpoint(const std::string& path,
                     const std::vector<Parameter>& params, int* epoch = nullptr,
                     float* metric_value = nullptr,
                     std::string* metric_name = nullptr,
                     std::string* config_text = nullptr);

bool load_checkpoint_metadata(const std::string& path,
                              CheckpointMetadata* metadata);
