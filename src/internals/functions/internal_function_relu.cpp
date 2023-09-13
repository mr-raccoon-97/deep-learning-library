#include "../config.h"
#include "../internal_array.hpp"
#include "../internal_tensor.hpp"
#include "internal_functions.hpp"

#if defined(USE_EIGEN_BACKEND)

#include <eigen3/Eigen/Dense>

namespace internal {

ReLU::ReLU(Tensor* input) : Function(input) {}

Tensor* ReLU::forward() {
    Tensor* result = input()->forward();
    Eigen::Map<Eigen::Array<scalar_type, 1, -1>> result_map(
        result->data(),
        result->size());

    result_map = result_map.cwiseMax(0);
    return result;
}

void ReLU::backward(Array* gradient) {
    if (input()->requires_gradient()) {
        Eigen::Map<Eigen::Array<scalar_type, 1, -1>> result_map(
            input()->data(),
            input()->size());

        Eigen::Map<Eigen::Array<scalar_type, 1, -1>> gradient_map(
            gradient->data(),
            gradient->size());

        Eigen::Array<scalar_type, 1, -1> mask = result_map > 0;
        gradient_map = gradient_map * mask;
        input()->backward(gradient);
    }
}

} // namespace internal

#endif // USE_EIGEN_BACKEND