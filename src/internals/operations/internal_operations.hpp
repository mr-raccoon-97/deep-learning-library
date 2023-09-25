/*************************************************************************************************\

This classes are non-leaf nodes representing operations. They are used to build the computational.
An operations should be implemented overriding the forward and backward methods, of the Tensor Base class.
The forward method should return the "this" pointer, and the backward method should pass the gradient
to the inputs.

For implementing operations use the data() method to access the data of the tensor and map it to some
data structure of some library. In this case I used Eigen::Map so avoid any unnecessary copy of the data
when performing the operations, and to be able to use the optimized Eigen library for the operations.

/*************************************************************************************************/


#ifndef INTERNAL_OPERATION_HPP
#define INTERNAL_OPERATION_HPP

#include "../internal_tensor.hpp"
#include "../internal_expression.hpp"

namespace internal {

class Operation : public Expression {
    public:
    Operation(Tensor* first, Tensor* second) {
        first_operand_ = first;
        second_operand_ = second;
        requires_gradient(first->requires_gradient() || second->requires_gradient());
    }

    Tensor* first_operand() const { return first_operand_; }
    Tensor* second_operand() const { return second_operand_; }
    
    private:
    Tensor* first_operand_;
    Tensor* second_operand_;
};

class Addition : public Operation {
    public:
    Addition(Tensor* first, Tensor* second);
    Tensor* forward() final;
    void backward(Tensor* gradient) const final;
};


class Multiplication : public Operation {
    public:
    Multiplication(Tensor* first, Tensor* second);
    Tensor* forward() final;
    void backward(Tensor* gradient) const final;
};

class Matmul : public Operation {
    public:
    Matmul(Tensor* first, Tensor* second);

    Tensor* forward() final;
    void backward(Tensor* gradient) const final;

    size_type rows_dimension() const { return first_operand()->shape().front(); }
    size_type inner_dimension() const { return  first_operand()->shape().back(); }
    size_type columns_dimension() const { return second_operand()->shape().back(); }
};

} // namespace internal

#endif // INTERNAL_OPERATION