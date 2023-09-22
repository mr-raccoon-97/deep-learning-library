#ifndef LAYERS_ABSTRACT_LAYER_H
#define LAYERS_ABSTRACT_LAYER_H

#include "../tensor.h"

namespace net::abstract {

class Layer {
    public:
    using size_type = Tensor::size_type;
    using shape_type = Tensor::shape_type;
    using scalar_type = Tensor::scalar_type;

    virtual ~Layer() = default;
    virtual Tensor forward(const Tensor& input) = 0;
};

} // namespace net::layer

#endif // LAYERS_ABSTRACT_LAYER_H