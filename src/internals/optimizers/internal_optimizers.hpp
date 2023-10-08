#ifndef INTERNAL_OPTIMIZERS_HPP
#define INTERNAL_OPTIMIZERS_HPP

#include <iostream>
#include <vector>

namespace internal {

class Tensor;

struct Optimizer {
    virtual ~Optimizer() = default;
    virtual void add_parameter(Tensor* parameter) = 0;
    virtual void step() = 0;
};

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