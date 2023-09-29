#pragma once
#include <iostream>
#include <vector>
#include <memory>

#include "initializers.h"
#include "statistics/distributions.h"

namespace internal { class Tensor; }

namespace net {

class Tensor {
    public:
    using scalar_type = float;
    using pointer = scalar_type*;
    using const_pointer = const scalar_type*;
    using size_type = std::size_t;
    using shape_type = std::vector<size_type>;
    using storage_type = std::vector<scalar_type>;
    using iterator = storage_type::iterator;
    using const_iterator = storage_type::const_iterator;

    Tensor() = default;
    Tensor(std::shared_ptr<internal::Tensor> tensor);
    Tensor(shape_type shape, bool gradient_requirement = false);
    Tensor(shape_type shape, requires_gradient gradient_requirement = requires_gradient::False);

    void reshape(shape_type shape);
    
    void backward(const Tensor& gradient);
    void perform();

    void fill(initializer distribution);
    void fill(scalar_type value);
    void fill(std::vector<scalar_type> values);

    internal::Tensor* internal() const;
    internal::Tensor* internal();

    iterator begin();
    iterator end();
    const_iterator begin() const;
    const_iterator end() const;
    const_iterator cbegin() const;
    const_iterator cend() const;

    Tensor gradient() const;

    pointer data();
    const_pointer data() const;
    shape_type shape() const;
    size_type rank() const;

    friend std::ostream& operator<<(std::ostream& ostream, const Tensor& tensor);

    private:
    std::shared_ptr<internal::Tensor> tensor_;
};

Tensor operator + (const Tensor& first, const Tensor& second);
Tensor operator * (const Tensor& first, const Tensor& second);
Tensor matmul(const Tensor& first, const Tensor& second);

} // namespace net