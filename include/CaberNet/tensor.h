#pragma once

#include <iostream>
#include <vector>
#include <memory>

#include "tensor/tensor_float32.h"
#include "tensor/tensor_int16.h"

namespace net {

template<typename T = float>
class Tensor {
    public:
    Tensor() { throw std::runtime_error("Bad type or not implemented type."); }
};

template<>
struct Tensor<float> : public TensorFloat32 {
    using TensorFloat32::TensorFloat32;
};

template<>
struct Tensor<int> : public TensorInt16 {
    using TensorInt16::TensorInt16;
};

std::ostream& operator<<(std::ostream& ostream, const Tensor<float>& tensor);
std::ostream& operator<<(std::ostream& ostream, const Tensor<int>& tensor);

Tensor<float> matmul(const Tensor<float>& first, const Tensor<float>& second);
Tensor<float> operator + (const Tensor<float>& first, const Tensor<float>& second);
Tensor<float> operator * (const Tensor<float>& first, const Tensor<float>& second);

} // namespace net