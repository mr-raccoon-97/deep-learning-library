#include "../config.h"
#include "../internal_tensor.hpp"
#include "../internal_array.hpp"
#include "./internal_operations.hpp"

#if defined(USE_EIGEN_BACKEND)

namespace internal {

void Tensor::multiply(const Tensor* other) {
    if(shape() != other->shape()) throw std::runtime_error("shape mismatch");
    Eigen::Map<Eigen::Array<scalar_type, 1, -1>> this_map(data(), size());
    Eigen::Map<const Eigen::Array<scalar_type, 1, -1>> other_map(other->data(), other->size());
    this_map *= other_map;
}

Multiplication::Multiplication(Tensor* first, Tensor* second)
:   Operation(first, second) {
    if(first->shape() != second->shape()) throw std::runtime_error("shape mismatch");
    reshape(first->shape());
}

Tensor* Multiplication::forward() {

    Eigen::Map<Eigen::Array<scalar_type, 1, -1>> this_map(
        this->data(),
        this->size() );

    Eigen::Map<const Eigen::Array<scalar_type, 1, -1>> first_operand_map(
        first_operand()->forward()->data(),
        first_operand()->size() );
        
    Eigen::Map<const Eigen::Array<scalar_type, 1, -1>> second_operand_map(
        second_operand()->forward()->data(),
        second_operand()->size() );

    this_map = first_operand_map * second_operand_map;
    return this;
}

void Multiplication::backward(Tensor* gradient) const {
    Eigen::Map<Eigen::Array<scalar_type, 1, -1>> gradient_map(gradient->data(), gradient->size());

    if (first_operand()->requires_gradient()) {
        Eigen::Map<const Eigen::Array<scalar_type, 1, -1>> second_operand_map(
            second_operand()->data(),
            second_operand()->size()
        );
        
        if (second_operand()->requires_gradient()) {
            Tensor* gradient_copy = new Tensor(gradient);
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