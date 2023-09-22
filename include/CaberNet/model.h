#pragma once

#include "tensor.h"

namespace net::abstract {

class Model {
    public:
    virtual ~Model() = default;
    virtual Tensor forward(const Tensor& input) = 0;
};

} // namespace net::abstract

