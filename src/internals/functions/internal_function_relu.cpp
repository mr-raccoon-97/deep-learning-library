#include "../config.h"
#include "../internal_array.hpp"
#include "../internal_tensor.hpp"
#include "internal_functions.hpp"

#if defined(USE_EIGEN_BACKEND)

#include <eigen3/Eigen/Dense>

namespace internal {

ReLU::ReLU(Tensor* input) : Function(input) {}

Tensor* ReLU::forward() {
    this->move(input()->forward());
    Eigen::Map<Eigen::Array<scalar_type, 1, -1>> this_map(
        this->data(),
        this->size());

    this_map = this_map.cwiseMax(0);
    return this;
}

void ReLU::backward(Array* gradient) const {
    if (requires_gradient()) {
        Eigen::Map<const Eigen::Array<scalar_type, 1, -1>> result_map(
            this->data(),
            this->size());

        Eigen::Map<Eigen::Array<scalar_type, 1, -1>> gradient_map(
            gradient->data(),
            gradient->size());

        gradient_map = gradient_map * (result_map > 0).cast<scalar_type>();
        this->backward(gradient);
    }
}

} // namespace internal

#endif // USE_EIGEN_BACKEND