#include "../config.h"
#include "../internal_tensor.hpp"
#include "../internal_array.hpp"
#include "internal_operation_multiplication.h"

#if defined(USE_EIGEN_BACKEND)

#include <eigen3/Eigen/Dense>

namespace internal {

Multiplication::Multiplication(const Tensor* first, const Tensor* second)
:   Operation(first, second) {
    if(first->shape() != second->shape()) throw std::runtime_error("shape mismatch");
}

std::unique_ptr<Tensor> Multiplication::perform() const {
    std::unique_ptr<Tensor> result = std::make_unique<Tensor>(first_operand()->shape());
    Eigen::Map<Eigen::Array<scalar_type, 1, -1>> result_map(result->data(), result->size());
    Eigen::Map<const Eigen::Array<scalar_type, 1, -1>> first_operand_map(first_operand()->data(), second_operand()->size());
    Eigen::Map<const Eigen::Array<scalar_type, 1, -1>> second_operand_map(second_operand()->data(), second_operand()->size());
    result_map = first_operand_map * second_operand_map;
    result->requires_gradient(gradient_requirement());
    result->is_leaf(false);
    result->derive_with(this);
    return result;
}

void Multiplication::backward(Array* gradient) const {
    Eigen::Map<Eigen::Array<scalar_type, 1, -1>> gradient_map(gradient->data(), gradient->size());

    if (first_operand()->requires_gradient()) {
        Eigen::Map<const Eigen::Array<scalar_type, 1, -1>> second_operand_map(
            second_operand()->data(),
            second_operand()->size()
        );
        
        if (second_operand()->requires_gradient()) {
            Array* gradient_copy = new Array(gradient);
            Eigen::Map<Eigen::Array<scalar_type, 1, -1>> gradient_copy_map(
                gradient_copy->data(),
                gradient_copy->size()
            );
            gradient_copy_map *= second_operand_map;
            first_operand()->backward(gradient_copy);
            delete gradient_copy;
        }
        
        else {
            gradient_map *= second_operand_map;
            first_operand()->backward(gradient);
        }
    }

    if (second_operand()->requires_gradient()) {
        Eigen::Map<const Eigen::Array<scalar_type, 1, -1>> first_operand_map(
            first_operand()->data(),
            first_operand()->size()
        );

        gradient_map *= first_operand_map;
        second_operand()->backward(gradient);
    }
}

} // namespace internal

#endif // USE_EIGEN_BACKEND