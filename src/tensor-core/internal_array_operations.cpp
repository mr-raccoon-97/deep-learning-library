#include "config.h"
#include "internal_array.hpp"

#if defined(USE_EIGEN_BACKEND)
#include <eigen3/Eigen/Dense>

namespace internal {
    
void Array::add(const Array* other) {
    if(shape_ != other->shape_) throw std::runtime_error("shape mismatch");
    Eigen::Map<Eigen::Array<scalar_type, 1, -1>> this_map(data(), size());
    Eigen::Map<const Eigen::Array<scalar_type, 1, -1>> other_map(other->data(), other->size());
    this_map += other_map;
}

void Array::multiply(const Array* other) {
    if(shape_ != other->shape_) throw std::runtime_error("shape mismatch");
    Eigen::Map<Eigen::Array<scalar_type, 1, -1>> this_map(data(), size());
    Eigen::Map<const Eigen::Array<scalar_type, 1, -1>> other_map(other->data(), other->size());
    this_map *= other_map;
}

} // namespace internal

#endif // USE_EIGEN_BACKEND