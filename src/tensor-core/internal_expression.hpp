#ifndef INTERNAL_EXPRESSION_HPP
#define INTERNAL_EXPRESSION_HPP

#include <iostream>
#include <vector>
#include <memory>

#include "internal_array.hpp"

namespace internal {

class Tensor;

struct Expression {
    virtual ~Expression() = default;
    virtual void backward(Array* gradient) const = 0;
    virtual Tensor perform() const = 0;
};

} // namespace internal

#endif // INTERNAL_EXPRESSION_HPP