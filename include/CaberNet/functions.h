#pragma once
#include "tensor.h"

namespace net::function {

Tensor<float> linear(const Tensor<float>& input, const Tensor<float>& weight, const Tensor<float>& bias);
Tensor<float> softmax(Tensor<float>& input, int axis);
Tensor<float> log_softmax(Tensor<float>&input, int axis);
Tensor<float> relu(const Tensor<float>& input);

} // namespace net::function
