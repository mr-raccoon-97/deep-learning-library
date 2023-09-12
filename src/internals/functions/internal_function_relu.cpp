#include "../config.h"
#include "../internal_types.h"
#include "../internal_array.hpp"
#include "../internal_tensor.hpp"

#include "internal_function_relu.h"

#if defined(USE_EIGEN_BACKEND)

#include <eigen3/Eigen/Dense>

namespace internal {

ReLU::ReLU(Tensor* input) : result_(input) {
    Eigen::Map<Eigen::Array<type::scalar_type, 1, -1>> result_map(
        result_->data(),
        result_->size());

    result_map = result_map.cwiseMax(0);
}

const Tensor* ReLU::result() const { return result_; }

void ReLU::backward(Array* gradient) const {
    if (result_->requires_gradient()) {
        Eigen::Map<Eigen::Array<type::scalar_type, 1, -1>> result_map(
            result_->data(),
            result_->size());

        Eigen::Map<Eigen::Array<type::scalar_type, 1, -1>> gradient_map(
            gradient->data(),
            gradient->size());

        Eigen::Array<type::scalar_type, 1, -1> mask = result_map > 0;
        gradient_map = gradient_map * mask;

        result_->backward(gradient);
    }
}

} // namespace internal

#endif // USE_EIGEN_BACKEND