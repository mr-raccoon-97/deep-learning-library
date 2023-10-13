#include "../config.h"
#include "../internal_tensor.hpp"

#include "internal_functions.hpp"

#if defined(USE_EIGEN_BACKEND)

namespace internal {

LogSoftmax::LogSoftmax(Tensor* input, int axis) : Function(input) {
    if (axis != 0 && axis != 1) { throw std::runtime_error("axis should be 0 or 1"); }
    axis_ = axis;
    reshape(input->shape());
}

Tensor* LogSoftmax::forward() {
    size_type rows = input()->shape().front();
    size_type columns = input()->size() / input()->shape().front();

    Eigen::Map<Eigen::Array<scalar_type, -1, -1, 1>> input_map(
        input()->forward()->data(),
        rows,
        columns );

    Eigen::Map<Eigen::Array<scalar_type, -1, -1, 1>> output_map(
        this->data(),
        rows,
        columns );


    if (axis_ == 0) {
        auto shifted = (input_map.colwise() - input_map.rowwise().maxCoeff());
        output_map = shifted.rowwise() - shifted.exp().colwise().sum().log();
    }

    else if (axis_ == 1) {        
        auto shifted = (input_map.colwise() - input_map.rowwise().maxCoeff());
        output_map = shifted.colwise() - shifted.exp().rowwise().sum().log();
    }

    else {
        throw std::runtime_error("axis should be 0 or 1");
    }

    return this;
}

void LogSoftmax::backward(Tensor* gradient) const {
    size_type rows = input()->shape().front();
    size_type columns = input()->size() / input()->shape().front();

    if (input()->requires_gradient()) {
        Eigen::Map<const Eigen::Array<scalar_type, -1, -1, 1>> output_map(
            this->data(),
            rows,
            columns );

        Eigen::Map<Eigen::Array<scalar_type, -1, -1, 1>> gradient_map(
            gradient->data(),
            rows,
            columns );

        if (axis_ == 0) {
            gradient_map -= output_map.exp().rowwise() * gradient_map.colwise().sum();
        }

        else if (axis_ == 1) {
            gradient_map -= output_map.exp().colwise() * gradient_map.rowwise().sum();
        }

        else {
            throw std::runtime_error("axis should be 0 or 1");
        }

        input()->backward(gradient);
    }
}

} // namespace internal

#endif // USE_EIGEN_BACKEND