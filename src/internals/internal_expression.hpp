/*************************************************************************************************\

This is just an interface class for the non-leaf nodes of the computational graph.

/*************************************************************************************************/

#ifndef INTERNAL_EXPRESSION_HPP
#define INTERNAL_EXPRESSION_HPP

#include "internal_tensor.hpp"

namespace internal {

class Expression : public Tensor {
    public:
    ~Expression() override = default;

    protected:
    Expression() 
    :   Tensor(false) {}

    Expression(shape_type shape, bool gradient_requirement = false)
    :   Tensor(shape, gradient_requirement, false) {}
};

} // namespace internal

#endif // INTERNAL_EXPRESSION_HPP