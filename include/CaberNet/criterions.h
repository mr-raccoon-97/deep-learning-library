/*#pragma once

#include "tensor.h"
#include "subscripts.h"

namespace internal { template<typename T> class Array; }

namespace net::base {

class Criterion {
    public:
    virtual ~Criterion() = default;
    Criterion(Tensor& output, Subscripts& targets);

    virtual void perform();
    virtual float loss();
    virtual void backward();

    private:
    std::shared_ptr<internal::Tensor> output_;
    std::shared_ptr<internal::Array<int>> targets_;
};

}


namespace net::criterion {

class NegativeLogLikelihood : public base::Criterion {
    public:
    NegativeLogLikelihood(Tensor& output, Subscripts& targets);
};

} // namespace net::criterion

*/