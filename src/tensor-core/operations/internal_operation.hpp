#ifndef INTERNAL_OPERATION_HPP
#define INTERNAL_OPERATION_HPP

#include <iostream>
#include <memory>
#include <vector>

#include "../internal_array.hpp"
#include "../internal_expression.hpp"
#include "../internal_tensor.hpp"

namespace internal {

class Operation : public Expression {
    public:
    using scalar_type = Tensor::scalar_type;
    using size_type = Tensor::size_type;
    using shape_type = Tensor::shape_type;

    ~Operation() override = default;

    Operation(const Tensor* first, const Tensor* second)
    :   operands_{ first, second }
    ,   gradient_requirement_(first->requires_gradient() || second->requires_gradient())
    {}
    
    bool gradient_requirement() const { return gradient_requirement_; }
    const Tensor* first_operand() const { return operands_.first; }
    const Tensor* second_operand() const { return operands_.second; }

    private:
    bool gradient_requirement_;
    std::pair<const Tensor*, const Tensor*> operands_;
};

} // namespace internal

#endif // INTERNAL_OPERATION