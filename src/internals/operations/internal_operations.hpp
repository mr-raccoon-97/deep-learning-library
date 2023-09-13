#ifndef INTERNAL_OPERATION_HPP
#define INTERNAL_OPERATION_HPP

#include "../internal_tensor.hpp"

namespace internal {

class Operation : public Tensor {
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
    void backward(Array* gradient) final;
};


class Multiplication : public Operation {
    public:
    Multiplication(Tensor* first, Tensor* second);
    Tensor* forward() final;
    void backward(Array* gradient) final;
};

class Matmul : public Operation {
    public:
    Matmul(Tensor* first, Tensor* second);

    Tensor* forward() final;
    void backward(Array* gradient) final;

    size_type rows_dimension() const { return first_operand()->shape().front(); }
    size_type inner_dimension() const { return  first_operand()->shape().back(); }
    size_type columns_dimension() const { return second_operand()->shape().back(); }
};

} // namespace internal

#endif // INTERNAL_OPERATION