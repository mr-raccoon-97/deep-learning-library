#include "../../include/CaberNet/tensor/tensor_float32.h"
#include "../../include/CaberNet/tensor.h"

#include "../internals/internal_tensor.hpp"
#include "../internals/internal_graph.hpp"
#include "../internals/operations/internal_operations.hpp"

namespace net {

TensorFloat32::TensorFloat32(std::shared_ptr<internal::Tensor> tensor) {
    tensor_ = tensor;
    internal::Graph::add(tensor_);
}

TensorFloat32::TensorFloat32(shape_type shape, bool gradient_requirement ) {
    tensor_ = std::make_shared<internal::Tensor>(shape);
    tensor_-> requires_gradient(gradient_requirement);
    internal::Graph::add(tensor_);
}

TensorFloat32::TensorFloat32(shape_type shape, requires_gradient gradient_requirement ) {
    tensor_ = std::make_shared<internal::Tensor>(shape);
    tensor_-> requires_gradient(static_cast<bool>(gradient_requirement));
    internal::Graph::add(tensor_);
}

void TensorFloat32::reshape(shape_type shape) {
    if(tensor_ == nullptr) tensor_ = std::make_shared<internal::Tensor>(shape, false, false);
    tensor_-> reshape(shape);
}

Tensor<float> TensorFloat32::gradient() const {
    Tensor<float> gradient = std::make_shared<internal::Tensor>(shape(), false);
    std::copy(tensor_->gradient()->begin(), tensor_->gradient()->end(), gradient.begin());
    return gradient;
}

internal::Tensor* TensorFloat32::internal() const {return tensor_.get(); }
internal::Tensor* TensorFloat32::internal() { return tensor_.get(); }

void TensorFloat32::backward(const Tensor<float>& gradient) { tensor_-> backward(gradient.internal()); }
void TensorFloat32::perform() { tensor_-> forward(); } // TODO : this should have a return type.

TensorFloat32::iterator TensorFloat32::begin() { return tensor_->begin(); }
TensorFloat32::iterator TensorFloat32::end() { return tensor_->end(); }
TensorFloat32::const_iterator TensorFloat32::begin() const { return tensor_->begin(); }
TensorFloat32::const_iterator TensorFloat32::end() const { return tensor_->end(); }
TensorFloat32::const_iterator TensorFloat32::cbegin() const { return tensor_->cbegin(); }
TensorFloat32::const_iterator TensorFloat32::cend() const { return tensor_->cend(); }

TensorFloat32::pointer TensorFloat32::data() { return tensor_->data(); }
TensorFloat32::const_pointer TensorFloat32::data() const { return tensor_->data(); }
TensorFloat32::shape_type TensorFloat32::shape() const { return tensor_->shape(); }
TensorFloat32::size_type TensorFloat32::rank() const { return tensor_->rank(); }

/*
Tensor operator + (const Tensor& first, const Tensor& second) {
    return Tensor(std::make_shared<internal::Addition>( first.internal(), second.internal() ));
}

Tensor operator * (const Tensor& first, const Tensor& second) {
    return Tensor(std::make_shared<internal::Multiplication>( first.internal(), second.internal() ));
}

Tensor matmul(const Tensor& first, const Tensor& second) {
    return Tensor(std::make_shared<internal::Matmul>( first.internal(), second.internal() ));
}
*/

void TensorFloat32::fill(initializer distribution) {
    distribution::Distribution<value_type>* filler = nullptr;
    switch (distribution) {
        case initializer::He :
            filler = new distribution::Normal<value_type>(0, std::sqrt(2.0 / shape().back()));
            for (auto& element : *this) element = filler->generate();
            break;
    
        default :
            throw std::runtime_error("Invalid initializer");
            break;
    }

    delete filler;
}

void TensorFloat32::fill(value_type value) {
    std::fill(tensor_->begin(), tensor_->end(), value);
}

void TensorFloat32::fill(std::vector<value_type> values) {
    std::move(values.begin(), values.end(), tensor_->begin());
}

} // namespace net

