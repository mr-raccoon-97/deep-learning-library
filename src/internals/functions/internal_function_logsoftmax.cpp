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
    this->copy(input()->forward());
    size_type rows = input()->shape().front();
    size_type columns = input()->size() / input()->shape().front();

    if (axis_ == 0) {
        Eigen::Map<Eigen::Array<scalar_type, -1, -1, 0>> input_map(
            this->data(),
            rows,
            columns );

        auto shifted = (input_map.colwise() - input_map.rowwise().maxCoeff());
        input_map = shifted.colwise() - shifted.exp().rowwise().sum().log();
    }

    else if (axis_ == 1) {        
        Eigen::Map<Eigen::Array<scalar_type, -1, -1, 0>> input_map(
            this->data(),
            rows,
            columns );

        auto shifted = (input_map.colwise() - input_map.rowwise().maxCoeff());
        input_map = shifted.colwise() - shifted.exp().rowwise().sum().log();
    }

    return this;
}

void LogSoftmax::backward(Tensor* gradient) const {
    if (input()->requires_gradient()) {
        throw std::runtime_error("Not implemented yet, if you want to contribute, you can start from here!");
        input()->backward(gradient);
    }
}

} // namespace internal

#endif // USE_EIGEN_BACKEND