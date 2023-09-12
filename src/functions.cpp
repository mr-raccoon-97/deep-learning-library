#include "../include/functions.h"
#include "internals/internal_tensor.hpp"
#include "internals/internal_expression.hpp"
#include "internals/internal_buffer.hpp"
#include "internals/functions/internal_function_linear.h"
#include "internals/functions/internal_function_softmax.h"
#include "internals/functions/internal_function_logsoftmax.h"
#include "internals/functions/internal_function_relu.h"

namespace net::function {

Tensor linear(const Tensor& input, const Tensor& weight, const Tensor& bias) {
    internal::Expression* expression = new internal::Linear(input.internal(), weight.internal(), bias.internal());
    internal::Buffer::cache(expression);
    Tensor result(expression->perform());
    return result;
}

// Task: add optional return type. 

void softmax(Tensor& input, int axis) {
    internal::Softmax::inplace(input.internal(), axis);
}

void log_softmax(Tensor& input, int axis) {
    internal::LogSoftmax::inplace(input.internal(), axis);
}

Tensor relu(const Tensor& input) {
    internal::Expression* expression = new internal::ReLU(input.internal());
    internal::Buffer::cache(expression);
}

} // namespace net::function