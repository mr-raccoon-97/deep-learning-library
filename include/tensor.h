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
    internal::Tensor* internal() {return _tensor.get(); }

    private:
    std::shared_ptr<internal::Tensor> _tensor;
};

} // namespace net


#endif // TENSOR_H