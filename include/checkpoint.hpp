#pragma once

#include <string>
#include <vector>

#include "module/module.hpp"

void save_checkpoint(const std::string& path,
                     const std::vector<Parameter>& params, int epoch,
                     float metric_value, const std::string& metric_name);

bool load_checkpoint(const std::string& path,
                     const std::vector<Parameter>& params, int* epoch = nullptr,
                     float* metric_value = nullptr,
                     std::string* metric_name = nullptr);
