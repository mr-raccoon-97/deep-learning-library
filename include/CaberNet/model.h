#pragma once

#include "tensor.h"

namespace net::base {

struct Model {
    using size_type = Tensor::size_type;
    using shape_type = Tensor::shape_type;

    virtual ~Model() = default;
    virtual Tensor forward(Tensor input) = 0;
};

} // namespace net::base

namespace net::layer {

struct Sequence : public base::Model {
    using Layer  = base::Model;
    std::vector<std::unique_ptr<Layer>> layers;
    Sequence() = default;
    Sequence(std::initializer_list<Layer*> layers) {
        for (auto layer : layers) this->layers.emplace_back(layer);
    }

    Sequence& operator=(std::initializer_list<Layer*> layers) {
        this->layers.clear();
        for (auto layer : layers) this->layers.emplace_back(layer);
        return *this;
    }

    Tensor forward(Tensor input) final {
        for (auto& layer : layers) input = layer->forward(input);
        return input;
    }
};

} // namespace net::layer