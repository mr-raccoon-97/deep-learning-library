#ifndef INTERNAL_OPTIMIZERS_HPP
#define INTERNAL_OPTIMIZERS_HPP

#include <iostream>
#include <vector>

namespace internal {

class Tensor;

class Optimizer {
    public:
    Optimizer() = default;
    virtual ~Optimizer() = default;
    virtual void add_parameter(Tensor* parameter) = 0;
    virtual void step() = 0;
};

class SGD : public Optimizer {
    public:
    SGD() = default;
    SGD(float learning_rate);
    ~SGD() final = default;

    void add_parameter(Tensor* parameter) final {
        parameters_.push_back(parameter);
    }

    void step() final;

    private:
    std::vector<Tensor*> parameters_;
    float learning_rate_;
};

}

#endif