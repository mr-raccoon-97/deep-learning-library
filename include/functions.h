#ifndef FUNCTIONS_H
#define FUNCTIONS_H

#include "tensor.h"

namespace net::function {

Tensor linear(const Tensor& input, const Tensor& weight, const Tensor& bias);
void softmax(Tensor& input, int axis);
void log_softmax(Tensor&input, int axis);

} // namespace net


#endif
