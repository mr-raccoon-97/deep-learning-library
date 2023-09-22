#ifndef LAYERS_DENSE_H
#define LAYERS_DENSE_H

#include <iostream>
#include <memory>

#include "layer.h"

namespace net::layer {

class Dense : public abstract::Layer {
    public:
    ~Dense() final = default;
    Dense(size_type input_features, size_type output_features) {
        throw std::runtime_error("We need to implement a statistics distribution module first");
    }

    Tensor forward(const Tensor& input) final {
        throw std::runtime_error("We need to implement a statistics distribution module first");
    }

    private:
    std::unique_ptr<Tensor> weight_;
    std::unique_ptr<Tensor> bias_;
};

} // namespace net

#endif // LAYERS_DENSE_HPP