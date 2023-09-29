#pragma once

#include <iostream>
#include <memory>

#include "tensor.h"
#include "subscripts.h"

namespace internal { 
    class Criterion;
}

namespace net::criterion {

class NegativeLogLikelihood {
    public:
    ~NegativeLogLikelihood();
    NegativeLogLikelihood(Tensor output, Subscripts targets);
    Tensor::scalar_type loss() const;

    private:
    std::unique_ptr<internal::Criterion> criterion_;
};

} // namespace net::criterion