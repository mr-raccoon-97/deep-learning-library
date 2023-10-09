#include "CaberNet/optimizers.h"

#include "internals/config.h"
#include "internals/internal_tensor.hpp"


#if defined(USE_EIGEN_BACKEND)

namespace net::optimizer {

void SGD::update(internal::Tensor* parameter) {
    Eigen::Map<Eigen::Array<internal::Tensor::scalar_type, 1, -1>> parameter_map(parameter->data(), parameter->size());
    Eigen::Map<Eigen::Array<internal::Tensor::scalar_type, 1, -1>> parameter_gradient_map(parameter->gradient()->data(), parameter->size());
    parameter_map -= learning_rate_ * parameter_gradient_map;
    parameter_gradient_map = 0;
}

}

#endif
