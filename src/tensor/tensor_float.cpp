#include "CaberNet/tensor/tensor_float.h"
#include "CaberNet/tensor.h"

#include "../internals/internal_tensor.hpp"
#include "../internals/internal_graph.hpp"
#include "../internals/operations/internal_operations.hpp"

namespace net {

TensorFloat::TensorFloat(std::shared_ptr<internal::Tensor> tensor) {
    tensor_ = tensor;
    internal::Graph::add(tensor_);
}

TensorFloat::TensorFloat(shape_type shape, bool gradient_requirement, bool detached ) {
    tensor_ = std::make_shared<internal::Tensor>(shape);
    tensor_-> requires_gradient(gradient_requirement);
    if(!detached) internal::Graph::add(tensor_);
}

TensorFloat::TensorFloat(shape_type shape, requires_gradient gradient_requirement , bool detached ) {
    tensor_ = std::make_shared<internal::Tensor>(shape);
    tensor_-> requires_gradient(static_cast<bool>(gradient_requirement));
    if(!detached) internal::Graph::add(tensor_);
}

void TensorFloat::reshape(shape_type shape) {
    if(tensor_ == nullptr) tensor_ = std::make_shared<internal::Tensor>(shape, false, false);
    tensor_-> reshape(shape);
}

Tensor<float> TensorFloat::gradient() const {
    Tensor<float> gradient = std::make_shared<internal::Tensor>(shape(), false);
    std::copy(tensor_->gradient()->begin(), tensor_->gradient()->end(), gradient.begin());
    return gradient;
}

internal::Tensor* TensorFloat::internal() const {return tensor_.get(); }
internal::Tensor* TensorFloat::internal() { return tensor_.get(); }

void TensorFloat::backward(const Tensor<float>& gradient) { tensor_-> backward(gradient.internal()); }
void TensorFloat::perform() { tensor_-> forward(); } // TODO : this should have a return type.

TensorFloat::iterator TensorFloat::begin() { return tensor_->begin(); }
TensorFloat::iterator TensorFloat::end() { return tensor_->end(); }
TensorFloat::const_iterator TensorFloat::begin() const { return tensor_->begin(); }
TensorFloat::const_iterator TensorFloat::end() const { return tensor_->end(); }
TensorFloat::const_iterator TensorFloat::cbegin() const { return tensor_->cbegin(); }
TensorFloat::const_iterator TensorFloat::cend() const { return tensor_->cend(); }

TensorFloat::pointer TensorFloat::data() { return tensor_->data(); }
TensorFloat::const_pointer TensorFloat::data() const { return tensor_->data(); }
TensorFloat::shape_type TensorFloat::shape() const { return tensor_->shape(); }
TensorFloat::size_type TensorFloat::rank() const { return tensor_->rank(); }

void TensorFloat::fill(initializer distribution) {
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

void TensorFloat::fill(value_type value) {
    std::fill(tensor_->begin(), tensor_->end(), value);
}

void TensorFloat::fill(std::vector<value_type> values) {
    std::move(values.begin(), values.end(), tensor_->begin());
}

} // namespace net

