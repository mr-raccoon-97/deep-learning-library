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

    BinaryExpression(const Tensor* first, const Tensor* second)
    :   operands{ first, second }
    ,   gradient_requirement(first->requires_gradient() || second->requires_gradient())
    {
        if (first->shape() != second->shape()) throw std::runtime_error("shape mismatch");
    }

    Tensor::shape_type shape() const { return operands.first->shape(); }

    virtual Tensor perform() const = 0;

    protected:
    std::pair<const Tensor*, const Tensor*> operands;
    bool gradient_requirement;
};

class Addition : public BinaryExpression {
    public:
    using BinaryExpression::BinaryExpression;

    ~Addition() final = default;

    Tensor perform() const final {
        Tensor result(operands.first);
        result.add(operands.second);
        result.requires_gradient(this->gradient_requirement);
        result.is_leaf(false);
        return result;
    }

    void backward(Array* gradient) final {
        Array* gradient_copy = new Array(gradient);
        if (operands.first->requires_gradient()) {
            operands.first->backward(gradient);
        }
        if (operands.second->requires_gradient()) {
            operands.second->backward(gradient_copy);
        }
        delete gradient_copy;
    }
};

class Multiplication : public BinaryExpression {
    public:
    using BinaryExpression::BinaryExpression;
    ~Multiplication() final = default;

    Tensor perform() const final {
        Tensor result(operands.first);
        result.multiply(operands.second);
        result.requires_gradient(this->gradient_requirement);
        result.is_leaf(false);
        return result;
    }

    void backward(Array* gradient) final {
        Array* gradient_copy = new Array(gradient);
        if (operands.first->requires_gradient()) {
            gradient->multiply(operands.second);
            operands.first->backward(gradient);
        }
        if (operands.second->requires_gradient()) {
            gradient_copy->multiply(operands.first);
            operands.second->backward(gradient_copy);
        }
        delete gradient_copy;
    }
};

} // namespace internal

#endif // INTERNAL_OPERATIONS