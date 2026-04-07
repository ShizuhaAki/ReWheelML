#pragma once

#include <vector>

#include "ndarray.hpp"

struct Parameter {
    NDArray* value;
    NDArray* grad;
};

class Module {
   public:
    virtual NDArray forward(const NDArray& x) = 0;
    virtual NDArray backward(const NDArray& grad_out) = 0;
    virtual void step(float lr) = 0;
    virtual ~Module() = default;
    virtual void train() {}
    virtual void eval() {}
    virtual std::vector<Parameter> parameters() { return {}; }
};
