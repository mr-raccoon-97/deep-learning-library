#include "../include/tensor.h"

#include "internals/internal_tensor.hpp"
#include "internals/internal_expression.hpp"
#include "internals/internal_buffer.hpp"
#include "internals/internal_array.hpp"

#include "internals/operations/internal_operation.hpp"
#include "internals/operations/internal_operation_addition.h"
#include "internals/operations/internal_operation_multiplication.h"
#include "internals/operations/internal_operation_matmul.h"

namespace net {

Tensor::Tensor(std::shared_ptr<internal::Tensor> tensor)
:   tensor_(tensor) {}

Tensor::Tensor(shape_type shape, bool gradient_requirement, bool node_status ) {
    tensor_ = std::make_shared<internal::Tensor>(shape);
    tensor_->requires_gradient(gradient_requirement);
    tensor_->is_leaf(node_status);
}

const internal::Tensor* Tensor::internal() const {return tensor_.get(); }
internal::Tensor* Tensor::internal() { return tensor_.get(); }

void Tensor::backward(const Tensor& gradient) {
    tensor_->backward(gradient.internal());
}

Tensor::iterator Tensor::begin() { return tensor_->begin(); }
Tensor::iterator Tensor::end() { return tensor_->end(); }
Tensor::const_iterator Tensor::begin() const { return tensor_->begin(); }
Tensor::const_iterator Tensor::end() const { return tensor_->end(); }
Tensor::const_iterator Tensor::cbegin() const { return tensor_->cbegin(); }
Tensor::const_iterator Tensor::cend() const { return tensor_->cend(); }

void Tensor::print_gradient() const {
    tensor_->print_gradient();
}

Tensor operator + (const Tensor& first, const Tensor& second) {
    internal::Expression* expression = new internal::Addition(first.internal(), second.internal());
    internal::Buffer::cache(expression);
    Tensor result(expression->perform());
    return result;
}

Tensor operator * (const Tensor& first, const Tensor& second) {
    internal::Expression* expression = new internal::Multiplication(first.internal(), second.internal());
    internal::Buffer::cache(expression);
    Tensor result(expression->perform());
    return result;
}

Tensor matmul(const Tensor& first, const Tensor& second) {
    internal::Expression* expression = new internal::Matmul(first.internal(), second.internal());
    internal::Buffer::cache(expression);
    Tensor result(expression->perform());
    return result;
}

} // namespace net

