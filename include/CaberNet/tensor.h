#pragma once

#include <iostream>
#include <vector>
#include <memory>

#include "tensor/tensor_float.h"
#include "tensor/tensor_int.h"

namespace net {

template<typename> class Tensor;

template<>
struct Tensor<float> : public TensorFloat {
    using TensorFloat::TensorFloat;
};

template<>
struct Tensor<int> : public TensorInt {
    using TensorInt::TensorInt;
};

std::ostream& operator<<(std::ostream& ostream, const Tensor<float>& tensor);
std::ostream& operator<<(std::ostream& ostream, const Tensor<int>& tensor);

Tensor<float> matmul(const Tensor<float>& first, const Tensor<float>& second);
Tensor<float> operator + (const Tensor<float>& first, const Tensor<float>& second);
Tensor<float> operator * (const Tensor<float>& first, const Tensor<float>& second);

} // namespace net