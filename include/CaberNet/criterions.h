#pragma once

#include <iostream>
#include <memory>

#include "tensor.h"

namespace internal { 
    class Criterion;
}

namespace net::criterion {

class NLLLoss {
    public:
    ~NLLLoss();
    NLLLoss(Tensor<float> output, Tensor<int> targets);
    float loss() const;
    void backward();

    private:
    std::unique_ptr<internal::Criterion> criterion_;
};

} // namespace net::criterion