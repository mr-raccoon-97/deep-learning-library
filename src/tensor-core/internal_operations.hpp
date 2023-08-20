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
    bool gradient_requirement;
    std::pair<const Tensor*, const Tensor*> operands;

    ~BinaryExpression() override = default;

    BinaryExpression(const Tensor* first, const Tensor* second)
    :   operands{ first, second }
    ,   gradient_requirement(first->requires_gradient() || second->requires_gradient())
    {}

    virtual Tensor perform() const = 0;
};

class Addition : public BinaryExpression {
    public:
    using BinaryExpression::BinaryExpression;
    ~Addition() final = default;
    Tensor perform() const final;
    void backward(Array* gradient) final;
};

class Multiplication : public BinaryExpression {
    public:
    using BinaryExpression::BinaryExpression;
    ~Multiplication() final = default;
    Tensor perform() const final;
    void backward(Array* gradient) final;
};

class MatrixMultiplication : public BinaryExpression {
    public:
    using scalar_type = Tensor::scalar_type;
    using size_type = Tensor::size_type;

    size_type rows;
    size_type columns;
    size_type inner_dimension;

    MatrixMultiplication(const Tensor* first, const Tensor* second);
    ~MatrixMultiplication() final = default;

    Tensor perform() const final ;
    void backward(Array* gradient) final ;
};

} // namespace internal

#endif // INTERNAL_OPERATIONS