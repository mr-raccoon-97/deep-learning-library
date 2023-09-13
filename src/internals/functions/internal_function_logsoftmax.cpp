#include "../config.h"
#include "../internal_tensor.hpp"

#include "internal_functions.hpp"

#if defined(USE_EIGEN_BACKEND)

#include <eigen3/Eigen/Dense>

namespace internal {

LogSoftmax::LogSoftmax(Tensor* input, int axis) : Function(input) {
    if (axis != 0 && axis != 1) { throw std::runtime_error("axis should be 0 or 1"); }
    axis_ = axis;
}

Tensor* LogSoftmax::forward() {

    size_type rows = input()->shape().front();
    size_type columns = input()->size() / input()->shape().front();

    if (axis_ == 0) {
        Eigen::Map<Eigen::Array<scalar_type, -1, -1, 0>> input_map(
            input()->forward()->data(),
            rows,
            columns );

        auto shifted = (input_map.colwise() - input_map.rowwise().maxCoeff());
        input_map = shifted.colwise() - shifted.exp().rowwise().sum().log();

    }

    else if (axis_ == 1) {        
        Eigen::Map<Eigen::Array<scalar_type, -1, -1, 0>> input_map(
            input()->forward()->data(),
            rows,
            columns );

        auto shifted = (input_map.colwise() - input_map.rowwise().maxCoeff());
        input_map = shifted.colwise() - shifted.exp().rowwise().sum().log();
    }
}

void LogSoftmax::backward(Array* gradient) const {
    if (input()->requires_gradient()) {
        input()->backward(gradient);
    }
}

} // namespace internal

#endif // USE_EIGEN_BACKEND