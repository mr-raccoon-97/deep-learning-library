#include "../config.h"
#include "../internal_types.h"
#include "../internal_tensor.hpp"
#include "../internal_array.hpp"
#include "internal_operation.hpp"
#include "internal_operation_matmul.h"

#if defined(USE_EIGEN_BACKEND)

#include <eigen3/Eigen/Dense>

namespace internal {

Matmul::Matmul(const Tensor* first, const Tensor* second)
:   Operation(first, second) {    
    if (first_operand()->rank() != 2 || second_operand()->rank() != 2)
        throw std::runtime_error("rank mismatch");

    if (first_operand()->shape().back() != second_operand()->shape().front())
        throw std::runtime_error("shape mismatch");
}

type::size_type Matmul::rows_dimension() const { return first_operand()->shape().front(); }
type::size_type Matmul::columns_dimension() const { return second_operand()->shape().back(); }
type::size_type Matmul::inner_dimension() const { return first_operand()->shape().back(); };

Tensor Matmul::perform() const {
    Tensor result({rows_dimension(), columns_dimension()});

    Eigen::Map<Eigen::Matrix<scalar_type, -1, -1, 1>> result_map(
        result.data(),
        rows_dimension(),
        columns_dimension() );

    Eigen::Map<const Eigen::Matrix<scalar_type, -1, -1, 1>> first_map(
        first_operand()->data(),
        rows_dimension(),
        inner_dimension() );

    Eigen::Map<const Eigen::Matrix<scalar_type, -1, -1, 0>> second_map(
        second_operand()->data(),
        inner_dimension(),
        columns_dimension() );
    
    result_map = first_map * second_map;
    result.requires_gradient(this->gradient_requirement());
    result.is_leaf(false);
    return result;
}

void Matmul::backward(Array* gradient) const {
    Eigen::Map<const Eigen::Matrix<scalar_type, -1, -1, 1>> row_gradient_map(
        gradient->data(),
        rows_dimension(),
        columns_dimension() );
    
    if (first_operand()->requires_gradient()) {
        Array* first_gradient = new Array({rows_dimension(), inner_dimension()});

        Eigen::Map<const Eigen::Matrix<scalar_type, -1, -1, 1>> second_map(
            second_operand()->data(),
            inner_dimension(),
            columns_dimension() );
        
        Eigen::Map<Eigen::Matrix<scalar_type, -1, -1, 1>> first_gradient_map(
            first_gradient->data(),
            rows_dimension(),
            inner_dimension() );

        first_gradient_map = row_gradient_map * second_map.transpose();
        first_operand()->backward(first_gradient);
        delete first_gradient;
    }
    
    Eigen::Map<const Eigen::Matrix<scalar_type, -1, -1, 0>> column_gradient_map(
        gradient->data(),
        rows_dimension(),
        columns_dimension() );

    if (second_operand()->requires_gradient()) {
        Array* second_gradient = new Array({inner_dimension(), columns_dimension()});
        Eigen::Map<const Eigen::Matrix<scalar_type, -1, -1, 0>> first_map(
            first_operand()->data(),
            rows_dimension(),
            inner_dimension() );

        Eigen::Map<Eigen::Matrix<scalar_type, -1, -1, 0>> second_gradient_map(
            second_gradient->data(),
            inner_dimension(),
            columns_dimension() );

        second_gradient_map = first_map.transpose() * column_gradient_map;
        second_operand()->backward(second_gradient);
        delete second_gradient;
    }
}

} // namespace internal

#endif // USE_EIGEN_BACKEND