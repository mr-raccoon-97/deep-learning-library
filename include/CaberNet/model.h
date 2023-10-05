#pragma once

#include "tensor.h"

namespace internal { class optimizer; }

namespace net {

template<class Derived>
class Model {
    public:
    using size_type = std::size_t;
    using shape_type = std::vector<size_t>;

    Tensor operator()(Tensor input) {
        return static_cast<Derived*>(this)->forward(input);
    }
};

} // namespace net