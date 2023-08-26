#include "../config.h"
#include "../internal_types.h"
#include "../internal_array.hpp"
#include "../internal_tensor.hpp"

#include "internal_function_linear.h"

#if defined(USE_EIGEN_BACKEND)

#include <eigen3/Eigen/Dense>

namespace internal {

Linear::Linear(const Tensor* input, const Tensor* weight, const Tensor* bias)
:   input_(input)
,   weight_(weight)
,   bias_(bias) {
    if (input->rank() != 2 || weight->rank() != 2) throw std::runtime_error("rank mismatch");
    if (input->shape().back() != weight->shape().front()) throw std::runtime_error("shape mismatch");
}

bool Linear::gradient_requirement() const {
    bool gradient_requirement = ( weight()->requires_gradient() || bias()->requires_gradient() );
    return ( gradient_requirement || input()->requires_gradient() );
}

type::size_type Linear::rows_dimension() const { return input()->shape().front(); }
type::size_type Linear::columns_dimension() const { return weight()->shape().back(); }
type::size_type Linear::inner_dimension() const { return input()->shape().back(); };

Tensor Linear::perform() const {
    Tensor result({rows_dimension(), columns_dimension()});

    Eigen::Map<Eigen::Matrix<type::scalar_type, -1, -1, 1>> result_map(
        result.data(),
        rows_dimension(),
        columns_dimension() );

    Eigen::Map<const Eigen::Matrix<type::scalar_type, -1, -1, 1>> input_map(
        input()->data(),
        rows_dimension(),
        inner_dimension() );

    Eigen::Map<const Eigen::Matrix<type::scalar_type, -1, -1, 0>> weight_map(
        weight()->data(),
        inner_dimension(),
        columns_dimension() );

    Eigen::Map<const Eigen::Matrix<type::scalar_type, 1, -1>> bias_map(
        bias()->data(),
        columns_dimension());

    result_map = (input_map * weight_map).rowwise() + bias_map;

    result.requires_gradient(gradient_requirement());
    result.is_leaf(false);
    return result;
}


void Linear::backward(Array* gradient) const {

    Eigen::Map<const Eigen::Matrix<type::scalar_type, -1, -1, 1>> row_gradient_map(
        gradient->data(),
        rows_dimension(),
        columns_dimension() );

    if (input()->requires_gradient()) {
        Array* input_gradient = new Array({rows_dimension(), inner_dimension()});

        Eigen::Map<const Eigen::Matrix<type::scalar_type, -1, -1, 1>> weight_map(
            weight()->data(),
            inner_dimension(),
            columns_dimension() );

        Eigen::Map<Eigen::Matrix<type::scalar_type, -1, -1, 1>> input_gradient_map(
            input_gradient->data(),
            rows_dimension(),
            inner_dimension() );

        input_gradient_map = row_gradient_map * weight_map.transpose();
        input()->backward(input_gradient);
        delete input_gradient;
    }
    
    Eigen::Map<const Eigen::Matrix<type::scalar_type, -1, -1, 0>> column_gradient_map(
        gradient->data(),
        rows_dimension(),
        columns_dimension() );

    if (weight()->requires_gradient()) {
        Array* weight_gradient = new Array({inner_dimension(), columns_dimension()});
        Eigen::Map<const Eigen::Matrix<type::scalar_type, -1, -1, 0>> input_map(
            input()->data(),
            rows_dimension(),
            inner_dimension() );

        Eigen::Map<Eigen::Matrix<type::scalar_type, -1, -1, 0>> weight_gradient_map(
            weight_gradient->data(),
            inner_dimension(),
            columns_dimension() );

        weight_gradient_map = input_map.transpose() * column_gradient_map;
        weight()->backward(weight_gradient);
        delete weight_gradient;
    }
    
    if (bias()->requires_gradient()) {
        Array* bias_gradient = new Array({columns_dimension()});
        Eigen::Map<Eigen::Matrix<type::scalar_type, 1, -1>> bias_gradient_map(
            bias_gradient->data(),
            columns_dimension() );

        bias_gradient_map = row_gradient_map.rowwise().sum();
        bias()->backward(bias_gradient);
        delete bias_gradient;
    }
}

#endif

} // namespace internal;