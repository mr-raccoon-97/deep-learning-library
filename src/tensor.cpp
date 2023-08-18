#include "../include/tensor.h"

#include "tensor-core/internal_tensor.hpp"
#include "tensor-core/internal_operations.hpp"
#include "tensor-core/internal_expression.hpp"
#include "tensor-core/internal_buffer.hpp"
#include "tensor-core/internal_array.hpp"

namespace net {

Tensor::Tensor(std::shared_ptr<internal::Tensor> tensor)
:   _tensor(tensor) {}

Tensor::Tensor(shape_type shape, bool requires_gradient, bool is_leaf ) {
    _tensor = std::make_shared<internal::Tensor>(shape, requires_gradient, is_leaf);
}

internal::Tensor* Tensor::internal() const {return _tensor.get(); }

Tensor::iterator Tensor::begin() { return _tensor->begin(); }
Tensor::iterator Tensor::end() { return _tensor->end(); }
Tensor::const_iterator Tensor::begin() const { return _tensor->begin(); }
Tensor::const_iterator Tensor::end() const { return _tensor->end(); }
Tensor::const_iterator Tensor::cbegin() const { return _tensor->cbegin(); }
Tensor::const_iterator Tensor::cend() const { return _tensor->cend(); }

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

