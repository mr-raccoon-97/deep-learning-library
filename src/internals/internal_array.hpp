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

namespace internal {

template<typename T>
class Array {
    public:
    using scalar_type = T;
    using pointer = scalar_type*;
    using const_pointer = const scalar_type*;

    using size_type = std::size_t;
    using shape_type = std::vector<size_type>;
    using storage_type = std::vector<scalar_type>;
    
    using iterator = typename storage_type::iterator;
    using const_iterator = typename storage_type::const_iterator;

    virtual ~Array() = default;

    Array() = default;
    Array(const Array* other) { copy(other); }
    Array(shape_type shape) { reshape(shape); }

    size_type size() const { return size_; }
    shape_type shape() const { return shape_; }
    size_type rank() const { return shape_.size(); }
    pointer data() { return storage_.data(); }
    const_pointer data() const { return storage_.data(); }

    iterator begin() { return storage_.begin(); }
    iterator end() { return storage_.end(); }
    const_iterator begin() const { return storage_.cbegin(); }
    const_iterator end() const { return storage_.cend(); }
    const_iterator cbegin() const { return storage_.cbegin(); }
    const_iterator cend() const { return storage_.cend(); }

    void copy(const Array* other) {
        size_ = other->size_;
        shape_ = other->shape_;
        storage_ = other->storage_;
    };

    void move(Array* other) {
        size_ = other->size_;
        shape_ = std::move(other->shape_);
        storage_ = std::move(other->storage_);
        other->size_ = 0;
        other->shape_.clear();
        other->storage_.clear();
    };

    void reshape(const shape_type& shape) {
        shape_ = shape;
        size_ = 1; for (size_type dimension : shape) size_ *= dimension;
        storage_.resize(size_);
    }

    void clear() {
        size_ = 0;
        shape_.clear();
        storage_.clear();
    }

    private:
    size_type size_;
    shape_type shape_;
    storage_type storage_;
};

} // namespace internal

#endif // INTERNAL_ARRAY_HPP