/*
This is the internal multidimensional array class that is used to store the data of the tensors.
It consists in a wrapper of a contiguous memory block (std::vector in this case, but it could be
changed into another container in the future) and a shape vector that stores the size of each
dimension of the array.
*/

#ifndef INTERNAL_ARRAY_HPP
#define INTERNAL_ARRAY_HPP

#include <iostream>
#include <vector>
#include <memory>

#include "internal_base.hpp"

namespace internal {

template<typename T>
class Array : public Base {
    public:
    using scalar_type = T;
    using pointer = scalar_type*;
    using const_pointer = const scalar_type*;

    using storage_type = std::vector<scalar_type>;    
    using iterator = typename storage_type::iterator;
    using const_iterator = typename storage_type::const_iterator;

    Array() = default;

    Array(const Array* other) : Base(other->shape()) {
        storage_ = other->storage_;
    }

    Array(shape_type shape) : Base(shape) {
        storage_.resize(size());
    }

    pointer data() { return storage_.data(); }
    const_pointer data() const { return storage_.data(); }

    iterator begin() { return storage_.begin(); }
    iterator end() { return storage_.end(); }
    const_iterator begin() const { return storage_.cbegin(); }
    const_iterator end() const { return storage_.cend(); }
    const_iterator cbegin() const { return storage_.cbegin(); }
    const_iterator cend() const { return storage_.cend(); }

    void copy(const Array* other) {
        reshape(other->shape());
        storage_ = other->storage_;
    };

    void move(Array* other) {
        reshape(other->shape());
        other->collapse();
        storage_ = std::move(other->storage_);
        other->storage_.clear();
    };

    void reshape(const shape_type& shape) {
        Base::reshape(shape);
        storage_.resize(size());
    }

    void clear() {
        storage_.clear();
        collapse();
    }

    private:
    storage_type storage_;
};

} // namespace internal

#endif // INTERNAL_ARRAY_HPP