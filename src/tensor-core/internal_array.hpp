#ifndef INTERNAL_ARRAY_HPP
#define INTERNAL_ARRAY_HPP

#include <iostream>
#include <vector>
#include <memory>

namespace internal {

class Array {
    public:
    using scalar_type = float;
    using pointer = scalar_type*;
    using const_pointer = const scalar_type*;
    using size_type = std::size_t;
    using shape_type = std::vector<size_type>;
    using storage_type = std::vector<scalar_type>;
    using iterator = storage_type::iterator;
    using const_iterator = storage_type::const_iterator;

    Array() = default;
    Array(shape_type shape)
    :   _shape(shape) {
        _size = 1; for (size_type dimension : shape) _size *= dimension;
        _storage.resize(_size);
    }

    size_type size() const { return _size; }
    shape_type shape() const { return _shape; }
    pointer data() { return _storage.data(); }
    const_pointer data() const { return _storage.data(); }

    iterator begin() { return _storage.begin(); }
    iterator end() { return _storage.end(); }
    const_iterator begin() const { return _storage.cbegin(); }
    const_iterator end() const { return _storage.cend(); }
    const_iterator cbegin() const { return _storage.cbegin(); }
    const_iterator cend() const { return _storage.cend(); }

    Array& add (const Array& other);
    Array& multiply (const Array& other);

    protected:
    void set_size(size_type size) { _size = size; }
    void set_shape(shape_type shape) { _shape = shape; }
    void resize_storage(size_type size) { _storage.resize(size); }

    private:
    size_type _size;
    shape_type _shape;
    storage_type _storage;
};

Array& Array::add (const Array& other) {
    if(_shape != other._shape) throw std::runtime_error("shape mismatch");
    for(size_type i = 0; i < _size; ++i) _storage[i] += other._storage[i];
    return *this;
}

Array& Array::multiply (const Array& other) {
    if(_shape != other._shape) throw std::runtime_error("shape mismatch");
    for(size_type i = 0; i < _size; ++i) _storage[i] *= other._storage[i];
    return *this;
}

} // namespace internal

#endif // INTERNAL_ARRAY_HPP