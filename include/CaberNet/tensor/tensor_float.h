#pragma once
#include <iostream>
#include <vector>
#include <memory>

#include "../initializers.h"
#include "../statistics/distributions.h"

namespace internal { class Tensor; }

namespace net {

template<typename T> class Tensor;

class TensorFloat {
    public:
    using value_type = float; // Needed for GMock's built-in matches
    using pointer = value_type*;
    using const_pointer = const value_type*;

    using size_type = std::size_t;
    using shape_type = std::vector<size_type>;
    using storage_type = std::vector<value_type>;

    using iterator = storage_type::iterator;
    using const_iterator = storage_type::const_iterator;

    TensorFloat() = default;
    TensorFloat(std::shared_ptr<internal::Tensor> tensor);
    TensorFloat(shape_type shape, bool gradient_requirement = false, bool detached = false);
    TensorFloat(shape_type shape, requires_gradient gradient_requirement,  bool detached = false);

    void reshape(shape_type shape);
    
    void backward(const Tensor<float>& gradient);
    void perform();

    void fill(initializer distribution);
    void fill(value_type value);
    void fill(std::vector<value_type> values);

    void copy(internal::Tensor* other);

    internal::Tensor* internal() const;
    internal::Tensor* internal();

    iterator begin();
    iterator end();
    const_iterator begin() const;
    const_iterator end() const;
    const_iterator cbegin() const;
    const_iterator cend() const;

    Tensor<float> gradient() const;

    pointer data();
    const_pointer data() const;
    shape_type shape() const;
    size_type rank() const;

    private:
    std::shared_ptr<internal::Tensor> tensor_;
};


} // namespace net
