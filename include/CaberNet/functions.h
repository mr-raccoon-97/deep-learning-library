#pragma once
#include "tensor.h"

namespace net::function {

Tensor linear(const Tensor& input, const Tensor& weight, const Tensor& bias);
Tensor softmax(Tensor& input, int axis);
Tensor log_softmax(Tensor&input, int axis);
Tensor relu(const Tensor& input);

} // namespace net::function
