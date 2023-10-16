#pragma once
#include <iostream>
#include <vector>
#include <memory>

namespace internal { template<typename T> class Array; }

namespace net {

template<typename T> class Tensor;

class TensorInt {
    public:
    using value_type = int;
    using pointer = value_type*;
    using const_pointer = const value_type*;

    using size_type = std::size_t;
    using shape_type = std::vector<size_type>;

    using iterator = std::vector<value_type>::iterator;
    using const_iterator = std::vector<value_type>::const_iterator;

    TensorInt() = default;
    TensorInt(std::shared_ptr<internal::Array<value_type>> subscripts);
    TensorInt(shape_type shape);

    void reshape(shape_type shape);
    void fill(value_type value);
    void fill(std::vector<value_type> values);

    void copy(internal::Array<value_type>* other);

    internal::Array<value_type>* internal() const;
    internal::Array<value_type>* internal();

    iterator begin();
    iterator end();
    const_iterator begin() const;
    const_iterator end() const;
    const_iterator cbegin() const;
    const_iterator cend() const;

    pointer data();
    const_pointer data() const;
    shape_type shape() const;
    size_type rank() const;

    private:
    std::shared_ptr<internal::Array<value_type>> data_;

};

} // namespace net