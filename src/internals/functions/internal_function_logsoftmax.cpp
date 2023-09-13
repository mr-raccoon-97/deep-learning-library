#include "../config.h"
#include "../internal_tensor.hpp"

#include "internal_functions.hpp"

#if defined(USE_EIGEN_BACKEND)

#include <eigen3/Eigen/Dense>

namespace internal {

void LogSoftmax::inplace(Tensor* input, int axis) {
    if (axis != 0 && axis != 1) { throw std::runtime_error("axis should be 0 or 1"); }

    size_type rows = input->shape().front();
    size_type columns = input->size() / input->shape().front();

    Eigen::Map<Eigen::Array<type::scalar_type, -1, -1, 0>> input_map;

    if (axis == 0) {
        input_map = Eigen::Map<Eigen::Array<type::scalar_type, -1, -1, 0>>(
            input->data(),
            rows,
            columns );
    }

    else if (axis == 1) {        
        input_map = Eigen::Map<Eigen::Array<type::scalar_type, -1, -1, 1>>(
            input->data(),
            rows,
            columns );
    }

    auto shifted = (input_map.colwise() - input_map.rowwise().maxCoeff());
    input_map = shifted.colwise() - shifted.exp().rowwise().sum().log();
}

} // namespace internal

#endif // USE_EIGEN_BACKEND