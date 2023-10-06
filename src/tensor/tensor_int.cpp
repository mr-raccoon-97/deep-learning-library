#include "CaberNet/tensor/tensor_int.h"
#include "CaberNet/tensor.h"

#include "../internals/internal_array.hpp"

namespace net {

TensorInt::TensorInt(std::shared_ptr<internal::Array<value_type>> subscripts) {
    data_ = subscripts;
}

TensorInt::TensorInt(shape_type shape) {
    data_ = std::make_shared<internal::Array<value_type>>(shape);
}

void TensorInt::reshape(shape_type shape) {
    data_->reshape(shape);
}

void TensorInt::fill(value_type value) {
    std::fill(data_->begin(), data_->end(), value);
}

void TensorInt::fill(std::vector<value_type> values) {
    std::move(values.begin(), values.end(), data_->begin());
}

internal::Array<TensorInt::value_type>* TensorInt::internal() const { return data_.get(); }
internal::Array<TensorInt::value_type>* TensorInt::internal() { return data_.get(); }

TensorInt::iterator TensorInt::begin() { return data_->begin(); }
TensorInt::iterator TensorInt::end() { return data_->end(); }
TensorInt::const_iterator TensorInt::begin() const { return data_->cbegin(); }
TensorInt::const_iterator TensorInt::end() const { return data_->cend(); }
TensorInt::const_iterator TensorInt::cbegin() const { return data_->cbegin(); }
TensorInt::const_iterator TensorInt::cend() const { return data_->cend(); }

TensorInt::pointer TensorInt::data() { return data_->data(); }
TensorInt::const_pointer TensorInt::data() const { return data_->data(); }
TensorInt::shape_type TensorInt::shape() const { return data_->shape(); }
TensorInt::size_type TensorInt::rank() const { return data_->rank(); }

} // namespace net