#pragma once

#include <iostream>
#include <vector>
#include <memory>

#include "tensor.h"

namespace internal { 

class Tensor; 
class Optimizer;
}

namespace net::base {

class Optimizer {
    public:
    void add_parameter(internal::Tensor* parameter);
    void step();

    internal::Optimizer* get() const { return optimizer_.get(); }

    protected:
    std::shared_ptr<internal::Optimizer> optimizer_ = nullptr;
};

}

namespace net::optimizer {

class SGD : public base::Optimizer {
    public: 
    ~SGD() = default;
    SGD(float learning_rate);
};

} // namespace net::optimizer



/*

SGD optimizer(0.1);
optimizer.add_parameter(...);
...
optimizer.update();

*/