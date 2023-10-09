#include "CaberNet/tensor.h"
#include "CaberNet/functions.h"
#include "internals/internal_tensor.hpp"
#include "internals/functions/internal_functions.hpp"

#include <iostream>
#include <memory>

namespace net::function {

Tensor<float> linear(const Tensor<float>& input, const Tensor<float>& weight, const Tensor<float>& bias) {
    return Tensor<float>(std::make_shared<internal::Linear>( input.internal(), weight.internal(), bias.internal() ));
}

Tensor<float> softmax(Tensor<float>& input, int axis) {
    return Tensor<float>(std::make_shared<internal::Softmax>( input.internal(), axis ));
}

Tensor<float> log_softmax(Tensor<float>& input, int axis) {
    return Tensor<float>(std::make_shared<internal::LogSoftmax>( input.internal(), axis ));
}

Tensor<float> relu(const Tensor<float>& input) {
    return Tensor<float>(std::make_shared<internal::ReLU>( input.internal() ));
}

} // namespace net::function