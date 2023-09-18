#include "../include/functions.h"
#include "internals/internal_tensor.hpp"
#include "internals/functions/internal_functions.hpp"

#include <iostream>
#include <memory>

namespace net::function {

Tensor linear(const Tensor& input, const Tensor& weight, const Tensor& bias) {
    return Tensor(std::make_shared<internal::Linear>( input.internal(), weight.internal(), bias.internal() ));
}

Tensor softmax(Tensor& input, int axis) {
    return Tensor(std::make_shared<internal::Softmax>( input.internal(), axis ));
}

Tensor log_softmax(Tensor& input, int axis) {
    return Tensor(std::make_shared<internal::LogSoftmax>( input.internal(), axis ));
}

Tensor relu(const Tensor& input) {
    return Tensor(std::make_shared<internal::ReLU>( input.internal() ));
}

} // namespace net::function