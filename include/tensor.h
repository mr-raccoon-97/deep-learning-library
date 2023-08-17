#ifndef TENSOR_H
#define TENSOR_H

#include <iostream>
#include <vector>
#include <memory>

namespace internal { class Tensor; }

namespace net {

class Tensor {
    public:
    using size_type = std::size_t;
    using shape_type = std::vector<size_type>;
    Tensor(std::shared_ptr<internal::Tensor> tensor);
    Tensor(shape_type shape, bool requires_gradient = true, bool is_leaf = true);
    internal::Tensor* internal() const;

    private:
    std::shared_ptr<internal::Tensor> _tensor;
};

Tensor operator + (const Tensor& first, const Tensor& second);
Tensor operator * (const Tensor& first, const Tensor& second);

} // namespace net


#endif // TENSOR_H