#include "../config.h"
#include "../internal_array.hpp"
#include "../internal_tensor.hpp"
#include "internal_operations.hpp"

#if defined(USE_EIGEN_BACKEND)

namespace internal {

Matmul::Matmul(Tensor* first, Tensor* second)
:   Operation(first, second) {    
    if (first_operand()->rank() != 2 || second_operand()->rank() != 2)
        throw std::runtime_error("rank mismatch");

    if (first_operand()->shape().back() != second_operand()->shape().front())
        throw std::runtime_error("shape mismatch");

    reshape({first_operand()->shape().front(), second_operand()->shape().back()});
}


Tensor* Matmul::forward() {

    Eigen::Map<Eigen::Matrix<scalar_type, -1, -1, 1>> this_map(
        this->data(),
        rows_dimension(),
        columns_dimension() );

    Eigen::Map<const Eigen::Matrix<scalar_type, -1, -1, 1>> first_operand_map(
        first_operand()->forward()->data(),
        rows_dimension(),
        inner_dimension() );

    Eigen::Map<const Eigen::Matrix<scalar_type, -1, -1, 1>> second_operand_map(
        second_operand()->forward()->data(),
        inner_dimension(),
        columns_dimension() );
    
    this_map = first_operand_map * second_operand_map;
    return this;
}


void Matmul::backward(Tensor* gradient) const {

    Eigen::Map<const Eigen::Matrix<scalar_type, -1, -1, 1>> gradient_map(
        gradient->data(),
        rows_dimension(),
        columns_dimension() );
    
    if (first_operand()->requires_gradient()) {
        Tensor* first_gradient = new Tensor({rows_dimension(), inner_dimension()}, false, false);

        Eigen::Map<const Eigen::Matrix<scalar_type, -1, -1, 1>> second_map(
            second_operand()->data(),
            inner_dimension(),
            columns_dimension() );
        
        Eigen::Map<Eigen::Matrix<scalar_type, -1, -1, 1>> first_gradient_map(
            first_gradient->data(),
            rows_dimension(),
            inner_dimension() );

        first_gradient_map = gradient_map * second_map.transpose();
        first_operand()->backward(first_gradient);
        delete first_gradient;
    }
    
    if (second_operand()->requires_gradient()) {
        Tensor* second_gradient = new Tensor({inner_dimension(), columns_dimension()}, false, false);
        Eigen::Map<const Eigen::Matrix<scalar_type, -1, -1, 1>> first_map(
            first_operand()->data(),
            rows_dimension(),
            inner_dimension() );

        Eigen::Map<Eigen::Matrix<scalar_type, -1, -1, 1>> second_gradient_map(
            second_gradient->data(),
            inner_dimension(),
            columns_dimension() );

        second_gradient_map = first_map.transpose() * gradient_map;
        second_operand()->backward(second_gradient);
        delete second_gradient;
    }
}

} // namespace internal

#endif // USE_EIGEN_BACKEND