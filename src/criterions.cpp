#include "../include/CaberNet/tensor.h"
#include "../include/CaberNet/subscripts.h"
#include "../include/CaberNet/criterions.h"

#include "internals/internal_tensor.hpp"
#include "internals/criterions/internal_criterions.hpp"

namespace net::criterion {

NegativeLogLikelihood::~NegativeLogLikelihood() = default;

NegativeLogLikelihood::NegativeLogLikelihood(Tensor output, Subscripts targets) {
    criterion_ = std::make_unique<internal::NLLLoss>(output.internal(), targets.internal());
}

Tensor::scalar_type NegativeLogLikelihood::loss() const {
    return criterion_->loss();
}

} // namespace net::criterion

