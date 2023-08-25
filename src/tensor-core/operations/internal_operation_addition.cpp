#include "../config.h"
#include "internal_operation_addition.h"

#if defined(USE_EIGEN_BACKEND)

#include <eigen3/Eigen/Dense>

namespace internal {

Addition::Addition(const Tensor* first, const Tensor* second)
:   Operation(first, second) {
    if(first->shape() != second->shape()) throw std::runtime_error("shape mismatch");
}


Tensor Addition::perform() const {
    Tensor result(first_operand()->shape());
    Eigen::Map<Eigen::Array<scalar_type, 1, -1>> result_map(result.data(), result.size());
    Eigen::Map<const Eigen::Array<scalar_type, 1, -1>> first_operand_map(first_operand()->data(), second_operand()->size());
    Eigen::Map<const Eigen::Array<scalar_type, 1, -1>> second_operand_map(second_operand()->data(), second_operand()->size());
    result_map = first_operand_map + second_operand_map;
    result.requires_gradient(gradient_requirement());
    result.is_leaf(false);
    return result;
}

void Addition::backward(Array* gradient) const {
    if (first_operand()->requires_gradient()) {
        if (second_operand()->requires_gradient()) {
            Array* gradient_copy = new Array(gradient);
            first_operand()->backward(gradient_copy);
            delete gradient_copy;
        }
        
        else {
            first_operand()->backward(gradient);
        }
    }

    if (second_operand()->requires_gradient()) {
        second_operand()->backward(gradient);
    }
}

#endif

} // namespace internal