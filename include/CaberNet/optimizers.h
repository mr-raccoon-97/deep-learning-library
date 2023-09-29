/*#pragma once

#include <iostream>
#include <vector>
#include <memory>

#include "tensor.h"

// Here we can break the design of pytorch again.
// PyTorch shuffles the data in the DataLoader class. But I think that is a bad design
// because then SGD will not be Stochastic, and it will be just a batch gradient descent.
// I think the optimizer should be responsible for shuffling the data. 
// Having concerns separated like this will allow us to have a functional prototype of
// the library working earlier. 

// Shuffling is a hard task, since we should see Tensors like a single block of memory like this:
//   xxxxxx|xxxxxx|xxxxxx|xxxxxx|xxxxxx|xxxxxx|xxxxxx|xxxxxx|xxxxxx|xxxxxx
// Where the first dimension is the batchsize. and each batch should be shuffled.
// This class should be responsible for shuffling the data.
// We can start first for a simple gradient descent and then try to implement a more complex
// optimizer with shuffling.

namespace internal { class Tensor; }

namespace net::base {

class Optimizer {
    public:
    Optimizer() = default;
    Optimizer(const std::vector<internal::Tensor*>& parameters);

    virtual void optimize() = 0;

    private:
    std::vector<internal::Tensor*> parameters_;
};

}

namespace net::optimizer {

class GradientDescent : public base::Optimizer {
    public:
    GradientDescent(const std::vector<internal::Tensor*>& parameters, float learning_rate = 0.01);

    void optimize() override;

    private:
    float learning_rate_;
};

}

*/