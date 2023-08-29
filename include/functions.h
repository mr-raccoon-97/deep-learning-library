#ifndef FUNCTIONS_H
#define FUNCTIONS_H

#include "tensor.h"

namespace net::function {

Tensor linear(const Tensor& input, const Tensor& weight, const Tensor& bias);

} // namespace net


#endif
