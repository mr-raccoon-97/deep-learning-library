#ifndef TENSOR_H
#define TENSOR_H

#include <iostream>
#include <vector>
#include <memory>

namespace internal { class Tensor; }

namespace net {

class Tensor {
    public:
    using scalar_type = float;
    using size_type = std::size_t;
    using shape_type = std::vector<size_type>;
    using storage_type = std::vector<scalar_type>;
    using iterator = storage_type::iterator;
    using const_iterator = storage_type::const_iterator;

    Tensor(std::shared_ptr<internal::Tensor> tensor);
    Tensor(shape_type shape, bool requires_gradient = true);

    void backward(const Tensor& gradient);

    const internal::Tensor* internal() const;
    internal::Tensor* internal();

    iterator begin();
    iterator end();
    const_iterator begin() const;
    const_iterator end() const;
    const_iterator cbegin() const;
    const_iterator cend() const;

    private:
    std::shared_ptr<internal::Tensor> tensor_;
};

Tensor operator + (const Tensor& first, const Tensor& second);
Tensor operator * (const Tensor& first, const Tensor& second);
Tensor matmul(const Tensor& first, const Tensor& second);

} // namespace net


#endif // TENSOR_H