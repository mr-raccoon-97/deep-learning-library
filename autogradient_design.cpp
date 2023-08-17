#include <iostream>
#include <vector>
#include <memory>

#include "src/tensor-core/internal_array.hpp"
#include "src/tensor-core/internal_expression.hpp"
#include "src/tensor-core/internal_tensor.hpp"
#include "src/tensor-core/internal_operations.hpp"
#include "src/tensor-core/internal_buffer.hpp"

namespace net {

class Tensor {
    public:
    using shape_type = internal::Array::shape_type;
    using size_type = internal::Array::size_type;

    Tensor(std::shared_ptr<internal::Tensor> tensor)
    :   tensor_(tensor) {}

    Tensor(shape_type shape, bool requires_gradient = true, bool is_leaf = true) {
        tensor_ = std::make_shared<internal::Tensor>(shape, requires_gradient, is_leaf);
    }

    void backward(const net::Tensor& gradient) const { tensor_->backward(gradient.internal()); }

    internal::Tensor* internal() const {return tensor_.get(); }

    auto begin() { return tensor_->begin(); }
    auto end() { return tensor_->end(); }
    auto cbegin() const { return tensor_->cbegin(); }
    auto cend() const { return tensor_->cend(); }

    private:
    std::shared_ptr<internal::Tensor> tensor_;
};


Tensor operator + (const Tensor& first, const Tensor& second) {
    internal::BinaryExpression* expression = new internal::Addition(first.internal(), second.internal());
    std::shared_ptr<internal::Tensor> internal_result = std::make_shared<internal::Tensor>(expression->perform());
    internal_result->derive_with(expression);
    Tensor result(std::move(internal_result));
    internal::Buffer::instance() << expression;
    return result;
}

Tensor operator * (const Tensor& first, const Tensor& second) {
    internal::BinaryExpression* expression = new internal::Multiplication(first.internal(), second.internal());
    std::shared_ptr<internal::Tensor> internal_result = std::make_shared<internal::Tensor>(expression->perform());
    internal_result->derive_with(expression);
    Tensor result(std::move(internal_result));
    internal::Buffer::instance() << expression;
    return result;
}

}

int main() {
    net::Tensor x({2, 2}, true, true);
    net::Tensor y({2, 2}, true, true);
    net::Tensor z({2, 2}, true, true);

    net::Tensor I({2, 2}, false, false);


    for (auto& element : x) element = 1;
    for (auto& element : y) element = -3;
    for (auto& element : z) element = 4;
    for (auto& element : I) element = 1;

    net::Tensor result({2, 2}, true, false);
    result = x * z + y * z;
    result.backward(I);

    x.internal()->print_gradient();
    std::cout << std::endl;
    y.internal()->print_gradient();
    std::cout << std::endl;
    z.internal()->print_gradient();
    std::cout << std::endl;
    return 0;
}