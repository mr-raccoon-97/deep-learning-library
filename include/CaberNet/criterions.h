#pragma once

#include <iostream>
#include <memory>

#include "tensor.h"

namespace internal { 
    class Criterion;
}

namespace net::criterion {

class NegativeLogLikelihood {
    public:
    ~NegativeLogLikelihood();
    NegativeLogLikelihood(Tensor<float> output, Tensor<int> targets);
    float loss() const;

    private:
    std::unique_ptr<internal::Criterion> criterion_;
};

} // namespace net::criterion