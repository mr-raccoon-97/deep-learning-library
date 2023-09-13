#include "../include/functions.h"
#include "internals/internal_buffer.hpp"
#include "internals/internal_tensor.hpp"
#include "internals/functions/internal_functions.hpp"

#include <iostream>
#include <memory>

namespace net::function {

Tensor linear(const Tensor& input, const Tensor& weight, const Tensor& bias) {
    std::shared_ptr<internal::Tensor> expression = std::make_shared<internal::Linear>(
        input.internal(),
        weight.internal(),
        bias.internal()
    );
    internal::Buffer::add(expression);
    return Tensor(expression);
}

void softmax(Tensor& input, int axis) {
    internal::Softmax::inplace(input.internal(), axis);
}

void log_softmax(Tensor& input, int axis) {
    internal::LogSoftmax::inplace(input.internal(), axis);
}

Tensor relu(const Tensor& input) {
    std::shared_ptr<internal::Tensor> expression = std::make_shared<internal::ReLU>(
        input.internal()
    );
    internal::Buffer::add(expression);
    return Tensor(expression);
}

} // namespace net::function