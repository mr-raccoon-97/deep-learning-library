#include "../config.h"
#include "../internal_tensor.hpp"
#include "./internal_functions.hpp"

#if defined(USE_EIGEN_BACKEND)

#include <eigen3/Eigen/Dense>

namespace internal {

Softmax::Softmax(Tensor* input, int axis) : Function(input) {
    if (axis != 0 && axis != 1) { throw std::runtime_error("axis should be 0 or 1"); }
    axis_ = axis;
}

Tensor* Softmax::forward() {    
    this->copy(input()->forward());

    size_type rows = input()->shape().front();
    size_type columns = input()->size() / input()->shape().front();

    if (axis_ == 0) {
        Eigen::Map<Eigen::Array<scalar_type, -1, -1, 0>> input_map(
            this->data(),
            rows,
            columns );


        auto shifted_exp = (input_map.colwise() - input_map.rowwise().maxCoeff()).exp();
        input_map = shifted_exp.colwise() / shifted_exp.rowwise().sum();
    }

    else if (axis_ == 1) {        
        Eigen::Map<Eigen::Array<scalar_type, -1, -1, 1>> input_map(
            this->data(),
            rows,
            columns );
        

        auto shifted_exp = (input_map.colwise() - input_map.rowwise().maxCoeff()).exp();
        input_map = shifted_exp.colwise() / shifted_exp.rowwise().sum();
    }

    return this;
}

void Softmax::backward(Array* gradient) const {
    if (requires_gradient()) {
        input()->backward(gradient);
    }
}

} // namespace internal

#endif // USE_EIGEN_BACKEND