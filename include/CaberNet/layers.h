#pragma once

#include "model.h"

#include <iostream>
#include <memory>
#include <vector>
#include <variant>

namespace net::layer {

class Linear : public Model<Linear> {
    public:
    Linear(
        size_type input_features,
        size_type output_features,
        initializer distribution = initializer::He );

    Tensor<float> forward(Tensor<float> x);

    private:
    Tensor<float> weight_;
    Tensor<float> bias_;
};

struct ReLU : public Model<ReLU> {
    ReLU() = default;
    Tensor<float> forward(Tensor<float> input);
};

struct Softmax : public Model<Softmax> {
    int axis;
    Softmax(int axis);
    Tensor<float> forward(Tensor<float> input);
};

struct LogSoftmax : public Model<LogSoftmax> {
    int axis;
    LogSoftmax(int axis);
    Tensor<float> forward(Tensor<float> input);
};


class Sequence : public Model<Sequence> {
    using layer_variant = std::variant<
        Linear,
        ReLU,
        Softmax,
        LogSoftmax
    >;
    public:


    template<class ... Layers>
    Sequence(Layers&& ... layers) {
        layers_ = { std::forward<Layers>(layers)... };
    }

    Tensor<float> forward(Tensor<float> input) {
        for (auto& layer : layers_) {
            input = std::visit([input](auto&& argument) { return argument.forward(input); }, layer);
        }
        return input;
    }


    private:
    std::vector<layer_variant> layers_;
};

} // namespace net