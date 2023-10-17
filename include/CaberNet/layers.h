#pragma once

#include "model.h"

#include <iostream>
#include <memory>
#include <vector>
#include <variant>

namespace internal {
    class Tensor;
};

namespace net::layer {

class Linear : public Model<Linear> {
    public:
    ~Linear();
    Linear(
        size_type input_features,
        size_type output_features,
        initializer distribution = initializer::He );

    Tensor<float> forward(Tensor<float> x);

    void set_optimizer(std::shared_ptr<net::base::Optimizer> optimizer);

    std::vector<internal::Tensor*> parameters() const {
        return { weight_.internal(), bias_.internal() };
    }
  
    private:
    Tensor<float> weight_;
    Tensor<float> bias_;
};

struct ReLU : public Model<ReLU> {
    ReLU() = default;
    
    std::vector<internal::Tensor*> parameters()const {
        return {};
    }

    Tensor<float> forward(Tensor<float> input);
    void set_optimizer(std::shared_ptr<net::base::Optimizer> optimizer) { return; }
};

struct Softmax : public Model<Softmax> {
    int axis;
    Softmax(int axis);

    std::vector<internal::Tensor*> parameters()const {
        return {};
    }

    Tensor<float> forward(Tensor<float> input);
    void set_optimizer(std::shared_ptr<net::base::Optimizer> optimizer) { return; }
};

struct LogSoftmax : public Model<LogSoftmax> {
    int axis;
    LogSoftmax(int axis);
    
    std::vector<internal::Tensor*> parameters()const {
        return {};
    }

    Tensor<float> forward(Tensor<float> input);
    void set_optimizer(std::shared_ptr<net::base::Optimizer> optimizer) { return; }
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

    std::vector<internal::Tensor*> parameters() const {

        std::vector<internal::Tensor*> parameter_list;
        for (auto& layer : layers_) {
            std::visit([&parameter_list](auto&& argument) {
                for(auto parameter : argument.parameters()) {
                    parameter_list.push_back(parameter);
                }
                
            }, layer);
        }

        return parameter_list;
    }

    void set_optimizer(std::shared_ptr<net::base::Optimizer> optimizer) {
        for (auto& layer : layers_) {
            std::visit([optimizer](auto&& argument) { argument.set_optimizer(optimizer); }, layer);
        }
    }

    private:
    std::vector<layer_variant> layers_;
};

} // namespace net