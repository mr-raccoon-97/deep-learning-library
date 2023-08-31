#include "../include/functions.h"

#include "tensor-internals/internal_tensor.hpp"
#include "tensor-internals/internal_expression.hpp"
#include "tensor-internals/internal_buffer.hpp"
#include "tensor-internals/functions/internal_function_linear.h"
#include "tensor-internals/functions/internal_functions_inplace.h"

namespace net::function {

Tensor linear(const Tensor& input, const Tensor& weight, const Tensor& bias) {
    internal::Expression* expression = new internal::Linear(input.internal(), weight.internal(), bias.internal());
    internal::Buffer::instance() << expression;
    Tensor result(expression->perform());
    return result;
}

void softmax(Tensor& input, int axis = 0) {
    internal::softmax_inplace(input.internal(), axis);
}

}