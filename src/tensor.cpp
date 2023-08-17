#include "../include/tensor.h"

#include "tensor-core/internal_tensor.hpp"
#include "tensor-core/internal_operations.hpp"
#include "tensor-core/internal_expression.hpp"
#include "tensor-core/internal_buffer.hpp"
#include "tensor-core/internal_array.hpp"

namespace net {

Tensor::Tensor(std::shared_ptr<internal::Tensor> tensor)
:   _tensor(tensor) {}

Tensor::Tensor(shape_type shape, bool requires_gradient = true, bool is_leaf = true) {
    _tensor = std::make_shared<internal::Tensor>(shape, requires_gradient, is_leaf);
}

internal::Tensor* Tensor::internal() const {return _tensor.get(); }

Tensor operator + (const Tensor& first, const Tensor& second) {
    internal::BinaryExpression* expression = new internal::Addition(first.internal(), second.internal());
    std::shared_ptr<internal::Tensor> internal_result = std::make_shared<internal::Tensor>(expression->perform());
    internal_result->derive_with(expression);
    Tensor result(std::move(internal_result));
    internal::Buffer::instance() << expression;
    return result;
}

Tensor operator * (const Tensor& first, const Tensor& second) {
    internal::BinaryExpression* expression = new internal::Multiplication(first.internal(), second.internal());
    std::shared_ptr<internal::Tensor> internal_result = std::make_shared<internal::Tensor>(expression->perform());
    internal_result->derive_with(expression);
    Tensor result(std::move(internal_result));
    internal::Buffer::instance() << expression;
    return result;
}

} // namespace net

