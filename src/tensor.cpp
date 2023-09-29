#include "../include/CaberNet/tensor.h"

#include "internals/internal_tensor.hpp"
#include "internals/internal_graph.hpp"
#include "internals/operations/internal_operations.hpp"

namespace net {

Tensor::Tensor(std::shared_ptr<internal::Tensor> tensor) {
    tensor_ = tensor;
    internal::Graph::add(tensor_);
}

Tensor::Tensor(shape_type shape, bool gradient_requirement ) {
    tensor_ = std::make_shared<internal::Tensor>(shape);
    tensor_-> requires_gradient(gradient_requirement);
    internal::Graph::add(tensor_);
}

Tensor::Tensor(shape_type shape, requires_gradient gradient_requirement ) {
    tensor_ = std::make_shared<internal::Tensor>(shape);
    tensor_-> requires_gradient(static_cast<bool>(gradient_requirement));
    internal::Graph::add(tensor_);
}

void Tensor::reshape(shape_type shape) {
    if(tensor_ == nullptr) tensor_ = std::make_shared<internal::Tensor>(shape, false, false);
    tensor_-> reshape(shape);
}

Tensor Tensor::gradient() const {
    Tensor gradient = std::make_shared<internal::Tensor>(shape(), false);
    std::copy(tensor_->gradient()->begin(), tensor_->gradient()->end(), gradient.begin());
    return gradient;
}

internal::Tensor* Tensor::internal() const {return tensor_.get(); }
internal::Tensor* Tensor::internal() { return tensor_.get(); }

void Tensor::backward(const Tensor& gradient) { tensor_-> backward(gradient.internal()); }
void Tensor::perform() { tensor_-> forward(); }

Tensor::iterator Tensor::begin() { return tensor_->begin(); }
Tensor::iterator Tensor::end() { return tensor_->end(); }
Tensor::const_iterator Tensor::begin() const { return tensor_->begin(); }
Tensor::const_iterator Tensor::end() const { return tensor_->end(); }
Tensor::const_iterator Tensor::cbegin() const { return tensor_->cbegin(); }
Tensor::const_iterator Tensor::cend() const { return tensor_->cend(); }

Tensor::pointer Tensor::data() { return tensor_->data(); }
Tensor::const_pointer Tensor::data() const { return tensor_->data(); }
Tensor::shape_type Tensor::shape() const { return tensor_->shape(); }
Tensor::size_type Tensor::rank() const { return tensor_->rank(); }

Tensor operator + (const Tensor& first, const Tensor& second) {
    return Tensor(std::make_shared<internal::Addition>( first.internal(), second.internal() ));
}

Tensor operator * (const Tensor& first, const Tensor& second) {
    return Tensor(std::make_shared<internal::Multiplication>( first.internal(), second.internal() ));
}

Tensor matmul(const Tensor& first, const Tensor& second) {
    return Tensor(std::make_shared<internal::Matmul>( first.internal(), second.internal() ));
}

void Tensor::fill(initializer distribution) {
    distribution::Distribution<scalar_type>* filler = nullptr;
    switch (distribution) {
        case initializer::He :
            filler = new distribution::Normal<scalar_type>(0, std::sqrt(2.0 / shape().back()));
            for (auto& element : *this) element = filler->generate();
            break;
    
        default :
            throw std::runtime_error("Invalid initializer");
            break;
    }

    delete filler;
}

void Tensor::fill(scalar_type value) {
    std::fill(tensor_->begin(), tensor_->end(), value);
}

void Tensor::fill(std::vector<scalar_type> values) {
    std::move(values.begin(), values.end(), tensor_->begin());
}

std::ostream& operator<<(std::ostream& ostream, const Tensor& tensor) {
    ostream << "[";
    for (auto element : tensor) ostream << element << ", ";
    ostream << "]";
    return ostream;
}

} // namespace net

