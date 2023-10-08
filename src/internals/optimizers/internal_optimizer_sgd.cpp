#include "../config.h"
#include "../internal_tensor.hpp"
#include "internal_optimizers.hpp"

#if defined(USE_EIGEN_BACKEND)

namespace internal {

SGD::SGD(float learning_rate) {
    learning_rate_ = learning_rate;
}

void SGD::update(Tensor* parameter) {
    Eigen::Map<Eigen::Array<Tensor::scalar_type, 1, -1>> parameter_map(parameter->data(), parameter->size());
    Eigen::Map<Eigen::Array<Tensor::scalar_type, 1, -1>> parameter_gradient_map(parameter->gradient()->data(), parameter->size());
    parameter_map -= learning_rate_ * parameter_gradient_map;
    parameter_gradient_map = 0;
}

}

#endif