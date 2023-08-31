#include "../config.h"
#include "../internal_tensor.hpp"
#include "internal_functions_inplace.h"
#include "../internal_types.h"

#if defined(USE_EIGEN_BACKEND)

#include <eigen3/Eigen/Dense>

namespace internal {

void softmax_inplace(Tensor* input, int axis) {
    if (axis == 0) {
        type::size_type rows = input->shape().front();
        type::size_type columns = input->size() / input->shape().front();
        Eigen::Map<Eigen::Array<type::scalar_type, -1, -1>> input_map(
            input->data(),
            rows,
            columns );

        input_map = (input_map.colwise() - input_map.rowwise().maxCoeff()).exp();
        input_map = input_map.colwise() / input_map.rowwise().sum();
    } else {
        throw std::runtime_error("axis should be 0 or 1");
    }    
}

} // namespace internal

#endif