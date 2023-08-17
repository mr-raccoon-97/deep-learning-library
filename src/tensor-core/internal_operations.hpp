#ifndef INTERNAL_OPERATIONS_HPP
#define INTERNAL_OPERATIONS_HPP

#include <iostream>
#include <memory>
#include <vector>

#include "internal_array.hpp"
#include "internal_expression.hpp"
#include "internal_tensor.hpp"

namespace internal {

class BinaryExpression : public Expression {
    public:
    ~BinaryExpression() override = default;
    BinaryExpression(const Tensor& first, const Tensor& second)
    :   operands{ first, second }
    ,   gradient_requirement(first.requires_gradient || second.requires_gradient)
    {
        if (first.shape() != second.shape()) throw std::runtime_error("shape mismatch");
    }
    Tensor::shape_type shape() const { return operands.first.shape(); }

    virtual Tensor perform() const = 0;

    protected:
    std::pair<const Tensor&, const Tensor&> operands;
    bool gradient_requirement;
};

class Addition : public BinaryExpression {
    public:
    using BinaryExpression::BinaryExpression;
    using BinaryExpression::backward;
    ~Addition() final = default;

    Tensor perform() const final {
        Tensor result(this->shape(), this->gradient_requirement, false);
        result.copy(operands.first);
        result.add(operands.second);
        return result;
    }

    void backward(Array& gradient) final {
        Array gradient_copy = gradient;
        if (operands.first.requires_gradient) {
            operands.first.backward(gradient);
        }
        if (operands.second.requires_gradient) {
            operands.second.backward(gradient_copy);
        }
    }
};

class Multiplication : public BinaryExpression {
    public:
    using BinaryExpression::BinaryExpression;
    using BinaryExpression::backward;
    ~Multiplication() final = default;

    Tensor perform() const final {
        Tensor result(this->shape(), this->gradient_requirement, false);
        result.copy(operands.first);
        result.multiply(operands.second);
        return result;
    }

    void backward(Array& gradient) {
        Array gradient_copy = gradient;
        if (operands.first.requires_gradient) {
            gradient.multiply(operands.second);
            operands.first.backward(gradient);
        }
        if (operands.second.requires_gradient) {
            gradient_copy.multiply(operands.first);
            operands.second.backward(gradient_copy);
        }
    }
};

} // namespace internal

#endif // INTERNAL_OPERATIONS