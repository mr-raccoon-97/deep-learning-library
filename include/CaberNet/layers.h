#pragma once

#include "model.h"

namespace net::layer {

struct Linear : public base::Model {
    Tensor weight;
    Tensor bias;
    Linear(size_type input_features, size_type output_features);
    Tensor forward(Tensor x) final;
};

struct ReLU : public base::Model {
    ~ReLU() final = default;
    ReLU() = default;
    Tensor forward(Tensor input) final;
};

struct Softmax : public base::Model {
    int axis;
    ~Softmax() final = default;
    Softmax(int axis);
    Tensor forward(Tensor input) final;
};

struct LogSoftmax : public base::Model {
    int axis;
    ~LogSoftmax() final = default;
    LogSoftmax(int axis);
    Tensor forward(Tensor input) final;
};

} // namespace net