#ifndef INTERNAL_EXPRESSION_HPP
#define INTERNAL_EXPRESSION_HPP

#include <iostream>
#include <vector>
#include <memory>

#include "internal_array.hpp"

namespace internal {

struct Expression {
    virtual ~Expression() = default;
    virtual void backward(Array& gradient) = 0;
};

} // namespace internal


#endif // INTERNAL_EXPRESSION_HPP