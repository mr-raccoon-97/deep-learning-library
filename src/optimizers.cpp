#include "CaberNet/optimizers.h"
#include "internals/optimizers/internal_optimizers.hpp"

namespace net::base {

Optimizer::~Optimizer() {}

internal::Optimizer* Optimizer::get() const {
    return optimizer_.get();
}

void Optimizer::add_parameter(internal::Tensor* parameter) {
    if(optimizer_) optimizer_->add_parameter(parameter);
}

void Optimizer::step() {
    if(optimizer_) optimizer_->step();
}

}

namespace net::optimizer {

SGD::~SGD() {}

SGD::SGD(float learning_rate) {
    optimizer_ = std::make_shared<internal::SGD>(learning_rate);
}

}
