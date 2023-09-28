#pragma once

#include <iostream>
#include <vector>
#include <memory>

#include "tensor.h"
#include "subscripts.h"

// Here we will break the design of pytorch and create something better. Since we have lazy evaluation, we can
// just store the pointers to the tensors and subscripts, and then evaluate the loss and pass the gradient
// without passing the output and the targets every time. So, when we implement a dataloader, we can just
// map the data into the inputs of the models and then perform the loss and backward pass. This will be
// more efficient than passing the output and the targets every time.


namespace internal {

class Tensor;
template<typename T> class Array;

}

namespace net {

namespace base {

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

namespace criterion {

class NegativeLogLikelihood : public base::Criterion {
    public:
    NegativeLogLikelihood(Tensor& output, Subscripts& targets);
};

class CrossEntropy {
 // ..
};

}



}