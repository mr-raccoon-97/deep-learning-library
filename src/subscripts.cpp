#include "../include/CaberNet/subscripts.h"

#include "internals/internal_array.hpp"

namespace net {

Subscripts::Subscripts(std::shared_ptr<internal::Array<int>> subscripts) {
    subscripts_ = subscripts;
}

Subscripts::Subscripts(shape_type shape) {
    subscripts_ = std::make_shared<internal::Array<int>>(shape);
}

void Subscripts::reshape(shape_type shape) {
    subscripts_->reshape(shape);
}

void Subscripts::fill(value_type value) {
    std::fill(subscripts_->begin(), subscripts_->end(), value);
}

void Subscripts::fill(std::vector<value_type> values) {
    std::move(values.begin(), values.end(), subscripts_->begin());
}

Subscripts::iterator Subscripts::begin() { return subscripts_->begin(); }
Subscripts::iterator Subscripts::end() { return subscripts_->end(); }
Subscripts::const_iterator Subscripts::begin() const { return subscripts_->cbegin(); }
Subscripts::const_iterator Subscripts::end() const { return subscripts_->cend(); }
Subscripts::const_iterator Subscripts::cbegin() const { return subscripts_->cbegin(); }
Subscripts::const_iterator Subscripts::cend() const { return subscripts_->cend(); }

Subscripts::pointer Subscripts::data() { return subscripts_->data(); }
Subscripts::const_pointer Subscripts::data() const { return subscripts_->data(); }
Subscripts::shape_type Subscripts::shape() const { return subscripts_->shape(); }
Subscripts::size_type Subscripts::rank() const { return subscripts_->rank(); }

std::ostream& operator<<(std::ostream& ostream, const Subscripts& subscript) {
    ostream << "[";
    for (auto element : subscript) ostream << element << ", ";
    ostream << "]";
    return ostream;
}

} // namespace net