#include <iostream>
#include <vector>
#include <memory>

namespace internal {

class Base {
    public:
    virtual ~Base() = default;
};

template<typename T>
class Array : Base {
    public:
    using scalar_type = T;
    using pointer = scalar_type*;
    using const_pointer = const scalar_type*;

    using storage_type = std::vector<scalar_type>;
    using iterator = typename storage_type::iterator;
    using const_iterator = typename storage_type::const_iterator;

    Array() = default;
    Array(std::size_t size) : data_(size) {
        for (std::size_t i = 0; i < size; ++i) data_[i] = 1;
    }

    storage_type data_;
};

class Tensor : public Array<float> {
    public:
    Tensor() = default;
    Tensor(std::size_t size) : Array(size) {}
};

};


namespace net {

class integer_32 {
    public:
    using scalar_type = int;
    using iterator = std::vector<scalar_type>::iterator;
    using const_iterator = std::vector<scalar_type>::const_iterator;

    integer_32(std::size_t size) {
        tensor_ = std::make_shared<internal::Array<scalar_type>>(size);
    }

    iterator begin() { return tensor_->data_.begin(); }
    iterator end() { return tensor_->data_.end(); }

    const_iterator begin() const { return tensor_->data_.cbegin(); }
    const_iterator end() const { return tensor_->data_.cend(); }

    private:
    std::shared_ptr<internal::Array<scalar_type>> tensor_;
};

class float_32 {
    public:
    using scalar_type = float;
    using iterator = std::vector<float>::iterator;
    using const_iterator = std::vector<float>::const_iterator;

    float_32(std::size_t size) {
        tensor_ = std::make_shared<internal::Tensor>(size);
    }

    iterator begin() { return tensor_->data_.begin(); }
    iterator end() { return tensor_->data_.end(); }

    const_iterator begin() const { return tensor_->data_.cbegin(); }
    const_iterator end() const { return tensor_->data_.cend(); }

    private:
    std::shared_ptr<internal::Tensor> tensor_;
};

template<typename T = float>
class Tensor{
    public:
    using scalar_type = T;

    private:
    std::shared_ptr<internal::Array<T>> data_;
};

template<>
class Tensor<float> : public float_32 {
    public:
    Tensor(std::size_t size) : float_32(size) {
        std::cout << "i'm a specialization";
    }
};

Tensor<float> fn(Tensor<float> x){
    return x;
}


} // namespace net

int main() {
    net::Tensor<float> tensor(10);
    net::Tensor tensor2 = net::fn(tensor);
    for(auto i : tensor2) std::cout << i << std::endl;
}