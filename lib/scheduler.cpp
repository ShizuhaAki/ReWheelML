#include "scheduler.hpp"

#include <algorithm>
#include <stdexcept>

ReduceLROnPlateau::ReduceLROnPlateau(Optimizer* optimizer, float factor,
                                     int patience, float min_lr,
                                     float threshold, const std::string& mode)
    : optimizer(optimizer),
      factor(factor),
      patience(patience),
      min_lr(min_lr),
      threshold(threshold),
      mode(mode),
      best_metric(0.0f),
      bad_epochs(0),
      has_best(false) {
    if (optimizer == nullptr) {
        throw std::invalid_argument("scheduler requires a valid optimizer");
    }
    if (factor <= 0.0f || factor >= 1.0f) {
        throw std::invalid_argument("scheduler factor must be in (0, 1)");
    }
    if (patience < 0) {
        throw std::invalid_argument("scheduler patience must be non-negative");
    }
    if (min_lr < 0.0f) {
        throw std::invalid_argument("scheduler min_lr must be non-negative");
    }
    if (mode != "min" && mode != "max") {
        throw std::invalid_argument("scheduler mode must be 'min' or 'max'");
    }
}

bool ReduceLROnPlateau::improved(float metric) const {
    if (mode == "min") {
        return metric < best_metric - threshold;
    }
    return metric > best_metric + threshold;
}

bool ReduceLROnPlateau::step(float metric) {
    if (!has_best) {
        best_metric = metric;
        has_best = true;
        bad_epochs = 0;
        return false;
    }

    if (improved(metric)) {
        best_metric = metric;
        bad_epochs = 0;
        return false;
    }

    ++bad_epochs;
    if (bad_epochs <= patience) {
        return false;
    }

    float current_lr = optimizer->get_lr();
    float new_lr = std::max(current_lr * factor, min_lr);
    bad_epochs = 0;
    if (new_lr >= current_lr) {
        return false;
    }

    optimizer->set_lr(new_lr);
    return true;
}
