#ifndef INTERNAL_OPTIMIZERS_HPP
#define INTERNAL_OPTIMIZERS_HPP

#include "CaberNet/optimizers.h"

#include <iostream>
#include <vector>

namespace internal {

class Tensor;

template<class Derived>
class OptimizerBase : public Optimizer {
    public:
    ~OptimizerBase() override = default;

    void add_parameter(Tensor* parameter) final {
        parameters_.push_back(parameter);
    }

    void step() final {
        for(Tensor* parameter : parameters_) {
            static_cast<Derived*>(this)->update(parameter);
        }
    }

    private:
    std::vector<Tensor*> parameters_;
};

class SGD : public OptimizerBase<SGD> {
    public:
    ~SGD() override = default;
    SGD(float learning_rate);
    void update(Tensor* parameter);

    private:
    float learning_rate_;
};

}

#endif