#pragma once

#include <string>

#include "optim/optim.hpp"

class Scheduler {
   public:
    virtual ~Scheduler() = default;
    virtual bool step(float metric) = 0;
};

class ReduceLROnPlateau : public Scheduler {
   private:
    Optimizer* optimizer;
    float factor;
    int patience;
    float min_lr;
    float threshold;
    std::string mode;
    float best_metric;
    int bad_epochs;
    bool has_best;

    bool improved(float metric) const;

   public:
    ReduceLROnPlateau(Optimizer* optimizer, float factor = 0.3f,
                      int patience = 3, float min_lr = 1e-5f,
                      float threshold = 1e-4f,
                      const std::string& mode = "min");

    bool step(float metric) override;
};
