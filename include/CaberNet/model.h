#pragma once

#include <iostream>
#include <variant>

#include "tensor.h"
#include "optimizers.h"

namespace net {

template<class Derived>
class Model {
    public:
    using size_type = std::size_t;
    using shape_type = std::vector<size_t>;

    Tensor<float> operator()(Tensor<float> input) {
        return static_cast<Derived*>(this)->forward(input);
    }

    void configure_optimizer(std::shared_ptr<net::base::Optimizer> instance) {
        static_cast<Derived*>(this)->set_optimizer(instance);
        optimizer_ = instance;
    }

    private:
    std::shared_ptr<net::base::Optimizer> optimizer_ = std::make_shared<net::optimizer::NoOptimization>();

    protected:
    Model() = default;
    Model(std::shared_ptr<net::base::Optimizer> optimizer) : optimizer_(optimizer) {
        static_cast<Derived*>(this)->set_optimizer(instance);
    }
};

} // namespace net
