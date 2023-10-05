#include "../../include/CaberNet/tensor/tensor_int16.h"
#include "../../include/CaberNet/tensor.h"

#include "../internals/internal_array.hpp"

namespace net {

TensorInt16::TensorInt16(std::shared_ptr<internal::Array<value_type>> subscripts) {
    data_ = subscripts;
}

TensorInt16::TensorInt16(shape_type shape) {
    data_ = std::make_shared<internal::Array<value_type>>(shape);
}

void TensorInt16::reshape(shape_type shape) {
    data_->reshape(shape);
}

void TensorInt16::fill(value_type value) {
    std::fill(data_->begin(), data_->end(), value);
}

void TensorInt16::fill(std::vector<value_type> values) {
    std::move(values.begin(), values.end(), data_->begin());
}

internal::Array<TensorInt16::value_type>* TensorInt16::internal() const { return data_.get(); }
internal::Array<TensorInt16::value_type>* TensorInt16::internal() { return data_.get(); }

TensorInt16::iterator TensorInt16::begin() { return data_->begin(); }
TensorInt16::iterator TensorInt16::end() { return data_->end(); }
TensorInt16::const_iterator TensorInt16::begin() const { return data_->cbegin(); }
TensorInt16::const_iterator TensorInt16::end() const { return data_->cend(); }
TensorInt16::const_iterator TensorInt16::cbegin() const { return data_->cbegin(); }
TensorInt16::const_iterator TensorInt16::cend() const { return data_->cend(); }

TensorInt16::pointer TensorInt16::data() { return data_->data(); }
TensorInt16::const_pointer TensorInt16::data() const { return data_->data(); }
TensorInt16::shape_type TensorInt16::shape() const { return data_->shape(); }
TensorInt16::size_type TensorInt16::rank() const { return data_->rank(); }

} // namespace net