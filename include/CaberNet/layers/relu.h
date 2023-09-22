#ifndef LAYERS_RELU_H
#define LAYERS_RELU_H

#include "layer.h"

namespace net::layer {

class ReLU : public abstract::Layer {
    public:
    ~ReLU() final = default;
    ReLU() = default;
    Tensor forward(const Tensor& input) final {
        throw std::runtime_error("We need to implement a statistics distribution module first");
    }
};

} // namespace net::layer

#endif // LAYERS_RELU_H