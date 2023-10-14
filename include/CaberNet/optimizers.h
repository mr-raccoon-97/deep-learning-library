#pragma once

#include <iostream>
#include <vector>
#include <memory>

#include "tensor.h"

namespace internal { 

class Tensor; 
struct Optimizer;

} // namespace internal

namespace net::base {

class Optimizer {
    public:
    ~Optimizer();
    void add_parameter(internal::Tensor* parameter);
    void step();

    internal::Optimizer* get() const;

    protected:
    Optimizer() = default;
    std::shared_ptr<internal::Optimizer> optimizer_ = nullptr;
};

}

namespace net::optimizer {

class SGD : public base::Optimizer {
    public:
    SGD() = default;
    ~SGD();
    SGD(float learning_rate);
};

} // namespace net::optimizer