#ifndef INTERNAL_EXPRESSION_HPP
#define INTERNAL_EXPRESSION_HPP

#include "internal_tensor.hpp"

namespace internal {

struct Expression : public Tensor {
    ~Expression() override = default;
};

} // namespace internal

#endif // INTERNAL_EXPRESSION_HPP