#pragma once

#include <iostream>
#include <memory>

#include "tensor.h"
#include "subscripts.h"

namespace internal { 
    template<typename T> class Array;
    class Criterion;
}

namespace net::base {

class Criterion {
    public:
    virtual ~Criterion() = default;
    virtual Tensor::scalar_type loss() const = 0;

    protected:
    Criterion() = default;
};

} // namespace net::base

namespace net::criterion {

class NegativeLogLikelihood : public base::Criterion {
    public:
    ~NegativeLogLikelihood() final;
    NegativeLogLikelihood(Tensor& output, Subscripts& targets);
    Tensor::scalar_type loss() const override;

    private:
    std::unique_ptr<internal::Criterion> criterion_;
};

} // namespace net::criterion