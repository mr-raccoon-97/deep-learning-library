#include "../include/tensor.h"

#include "tensor-core/internal_tensor.hpp"
#include "tensor-core/internal_expression.hpp"
#include "tensor-core/internal_buffer.hpp"
#include "tensor-core/internal_array.hpp"
#include "tensor-core/operations/internal_operation.hpp"
#include "tensor-core/operations/internal_operation_addition.h"
#include "tensor-core/operations/internal_operation_multiplication.h"
#include "tensor-core/operations/internal_operation_matmul.h"


namespace net {

Tensor::Tensor(std::shared_ptr<internal::Tensor> tensor)
:   tensor_(tensor) {}

Tensor::Tensor(shape_type shape, bool gradient_requirement, bool node_status ) {
    tensor_ = std::make_shared<internal::Tensor>(shape);
    tensor_->requires_gradient(gradient_requirement);
    tensor_->is_leaf(node_status);
}

internal::Tensor* Tensor::internal() const {return tensor_.get(); }

Tensor::iterator Tensor::begin() { return tensor_->begin(); }
Tensor::iterator Tensor::end() { return tensor_->end(); }
Tensor::const_iterator Tensor::begin() const { return tensor_->begin(); }
Tensor::const_iterator Tensor::end() const { return tensor_->end(); }
Tensor::const_iterator Tensor::cbegin() const { return tensor_->cbegin(); }
Tensor::const_iterator Tensor::cend() const { return tensor_->cend(); }

Tensor operator + (const Tensor& first, const Tensor& second) {
    internal::Expression* expression = new internal::Addition(first.internal(), second.internal());
    std::shared_ptr<internal::Tensor> internal_result = std::make_shared<internal::Tensor>(expression->perform());
    internal_result->derive_with(expression);
    Tensor result(std::move(internal_result));
    internal::Buffer::instance() << expression;
    return result;
}

Tensor operator * (const Tensor& first, const Tensor& second) {
    internal::Expression* expression = new internal::Multiplication(first.internal(), second.internal());
    std::shared_ptr<internal::Tensor> internal_result = std::make_shared<internal::Tensor>(expression->perform());
    internal_result->derive_with(expression);
    Tensor result(std::move(internal_result));
    internal::Buffer::instance() << expression;
    return result;
}

} // namespace net

