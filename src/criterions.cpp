#include "CaberNet/tensor.h"
#include "CaberNet/criterions.h"

#include "internals/internal_tensor.hpp"
#include "internals/criterions/internal_criterions.hpp"

namespace net::criterion {

NLLLoss::~NLLLoss() = default;

NLLLoss::NLLLoss(Tensor<float> output, Tensor<int> targets) {
    criterion_ = std::make_unique<internal::NLLLoss>(output.internal(), targets.internal());
}

float NLLLoss::loss() const {
    return criterion_->loss();
}

void NLLLoss::backward() {
    criterion_->backward();
}

} // namespace net::criterion

