#include "CaberNet/tensor.h"
#include "CaberNet/criterions.h"

#include "internals/internal_tensor.hpp"
#include "internals/criterions/internal_criterions.hpp"

namespace net::criterion {

NegativeLogLikelihood::~NegativeLogLikelihood() = default;

NegativeLogLikelihood::NegativeLogLikelihood(Tensor<float> output, Tensor<int> targets) {
    criterion_ = std::make_unique<internal::NLLLoss>(output.internal(), targets.internal());
}

float NegativeLogLikelihood::loss() const {
    return criterion_->loss();
}

} // namespace net::criterion

