#include "../include/CaberNet/tensor.h"
#include "../include/CaberNet/layers.h"

#include "internals/functions/internal_functions.hpp"
#include "internals/optimizers/internal_optimizers.hpp"


namespace net::layer {

/// constructors

Linear::~Linear() = default;

Linear::Linear(size_type input_features, size_type output_features, initializer distribution)
:   weight_(shape_type{output_features, input_features}),
    bias_(shape_type{1, output_features}, 0.0) {
    weight_.fill(distribution);
    bias_.fill(0.0);
}

Softmax::Softmax(int axis) : axis(axis) {}
LogSoftmax::LogSoftmax(int axis) : axis(axis) {}

/// settings

void Linear::set_optimizer(internal::Optimizer* optimizer) {
    optimizer->add_parameter(weight_.internal());
    optimizer->add_parameter(bias_.internal());
}

/// forward methods

Tensor<float> Linear::forward(Tensor<float> input) {
    return Tensor<float>(std::make_shared<internal::Linear>(input.internal(), weight_.internal(), bias_.internal()));

}

Tensor<float> ReLU::forward(Tensor<float> input) {
    return Tensor<float>(std::make_shared<internal::ReLU>(input.internal()));
}

Tensor<float> Softmax::forward(Tensor<float> input) {
    return Tensor<float>(std::make_shared<internal::Softmax>(input.internal(), axis));
}

Tensor<float> LogSoftmax::forward(Tensor<float> input) {
    return Tensor<float>(std::make_shared<internal::LogSoftmax>(input.internal(), axis));
}

} // namespace net::layer