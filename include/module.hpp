#pragma once
#include "ndarray.hpp"
class Module {
   public:
    virtual NDArray forward(const NDArray& x) = 0;
    virtual NDArray backward(const NDArray& grad_out) = 0;
    virtual void step(double lr) = 0;
    virtual ~Module() = default;
};