#pragma once

#include <iostream>
#include <vector>
#include <memory>


namespace internal { class Tensor; }

namespace net::base {

struct Optimizer {
    virtual ~Optimizer() = default;
    virtual void add_parameter(internal::Tensor* parameter) = 0;
    virtual void add_parameter(const std::vector<internal::Tensor*>& parameters) = 0;
    virtual void step() = 0;
};

template<class Derived>
class Optimize : public Optimizer {
    public:
    ~Optimize() override = default;

    void add_parameter(internal::Tensor* parameter) override final {
        parameters_.push_back(parameter);
    }

    void add_parameter(const std::vector<internal::Tensor*>& parameters) override final {
        parameters_.insert(parameters_.end(), parameters.begin(), parameters.end());
    }

    void step() override final {
        for(internal::Tensor* parameter : parameters_) {
            static_cast<Derived*>(this)->update(parameter);
        }
    }

    private:
    std::vector<internal::Tensor*> parameters_;
};

}

namespace net::optimizer {

class NoOptimization : public base::Optimize<NoOptimization> {
    public: 
    ~NoOptimization() = default;
    void update(internal::Tensor* parameter) {return;}
};


class SGD : public base::Optimize<SGD> {
    public: 
    SGD(float learning_rate): learning_rate_{learning_rate} {}
    ~SGD() = default;

    void update(internal::Tensor* parameter);

    protected:
    const float learning_rate_;
};

} // namespace net::optimizer