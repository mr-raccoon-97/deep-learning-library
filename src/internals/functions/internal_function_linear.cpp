#include "../config.h"
#include "../internal_array.hpp"
#include "../internal_tensor.hpp"
#include "internal_functions.hpp"

#if defined(USE_EIGEN_BACKEND)

// matmul(x, W.T) + b like pytorch

namespace internal {

Linear::Linear(Tensor* input, Tensor* weight, Tensor* bias)
:   Function(input)
,   weight_(weight)
,   bias_(bias) {
    if (input->rank() != 2 || weight->rank() != 2) throw std::runtime_error("rank mismatch");
    if (input->shape().back() != weight->shape().back()) throw std::runtime_error("shape mismatch between input and weight");
    if (bias->shape().back() != weight->shape().front()) throw std::runtime_error("shape mismatch between bias and weight");
    reshape({input->shape().front(), weight->shape().front()});
    bool gradient_requirement = input->requires_gradient() || ( weight->requires_gradient() || bias->requires_gradient() );
    requires_gradient(gradient_requirement);
}

Tensor* Linear::forward() {

    Eigen::Map<Eigen::Matrix<scalar_type, -1, -1, 1>> this_map(
        this->data(),
        rows_dimension(),
        columns_dimension());

    Eigen::Map<const Eigen::Matrix<scalar_type, -1, -1, 1>> input_map(
        input()->forward()->data(),
        rows_dimension(),
        inner_dimension() );

    Eigen::Map<const Eigen::Matrix<scalar_type, -1, -1, 1>> weight_map(
        weight()->forward()->data(),
        columns_dimension(),
        inner_dimension() );

    Eigen::Map<const Eigen::Matrix<scalar_type, 1, -1, 1>> bias_map(
        bias()->forward()->data(),
        columns_dimension());

    this_map = (input_map * weight_map.transpose()).rowwise() + bias_map;
    return this;
}


void Linear::backward(Array* gradient) const {
    
    Eigen::Map<const Eigen::Matrix<scalar_type, -1, -1, 1>> gradient_map(
        gradient->data(),
        rows_dimension(),
        columns_dimension() );

    if (input()->requires_gradient()) {
        Array* input_gradient = new Array({rows_dimension(), inner_dimension()});

        Eigen::Map<const Eigen::Matrix<scalar_type, -1, -1, 1>> weight_map(
            weight()->data(),
            columns_dimension(),
            inner_dimension() );

        Eigen::Map<Eigen::Matrix<scalar_type, -1, -1, 1>> input_gradient_map(
            input_gradient->data(),
            rows_dimension(),
            inner_dimension() );

        input_gradient_map = gradient_map * weight_map;
        input()->backward(input_gradient);
        delete input_gradient;
    }

    if (weight()->requires_gradient()) {
        Array* weight_gradient = new Array({columns_dimension(), inner_dimension()});
        Eigen::Map<const Eigen::Matrix<scalar_type, -1, -1, 1>> input_map(
            input()->data(),
            rows_dimension(),
            inner_dimension() );

        Eigen::Map<Eigen::Matrix<scalar_type, -1, -1, 1>> weight_gradient_map(
            weight_gradient->data(),
            columns_dimension(),
            inner_dimension() );

        weight_gradient_map = gradient_map.transpose() * input_map;
        weight()->backward(weight_gradient);
        delete weight_gradient;
    }

    if (bias()->requires_gradient()) {
        Array* bias_gradient = new Array({1,columns_dimension()});
        Eigen::Map<Eigen::Matrix<scalar_type, 1, -1, 1>> bias_gradient_map(
            bias_gradient->data(),
            columns_dimension() );
        bias_gradient_map = gradient_map.colwise().sum();
        bias()->backward(bias_gradient);
        delete bias_gradient;
    }
}

} // namespace internal

#endif // USE_EIGEN_BACKEND