#include <iostream>
#include <vector>
#include <memory>

//////////////////////////////

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

struct Expression {
    virtual ~Expression() = default;
    virtual void backward(Array& gradient) = 0;
};

class Tensor : public Array {
    public:


    Tensor(const Tensor& other) { copy(other); }
    Tensor(Tensor&& other) { move(std::move(other)); }
    Tensor& operator=(const Tensor& other) { if (this != &other) copy(other);  return *this; }
    Tensor& operator=(Tensor&& other) { if (this != &other) move(std::move(other)); return *this; }
    ~Tensor() { if (_requires_gradient) delete _gradient; }

    Tensor(shape_type shape, bool requires_gradient, bool is_leaf)
    :   Array(shape)
    ,   _requires_gradient(requires_gradient)
    ,   _is_leaf(is_leaf) {
        if (_requires_gradient) { 
            _gradient = new Array(shape);
        }
    }
    
    bool requires_gradient() const { return _requires_gradient; }
    bool is_leaf() const { return _is_leaf; }

    void derive_with(Expression* expression) { _expression_view = expression; }

    void backward(Array& gradient) const {
        if (_is_leaf) { _gradient->add(gradient); } 
        else { _expression_view->backward(gradient); }
    }

    void print_gradient() {
        if (_requires_gradient) {
            for(auto i = 0; i < _gradient->size(); ++i) std::cout << _gradient->data()[i] << " ";
        }
    }

    protected:

    void copy(const Tensor& other) {
        set_size(other.size());
        set_shape(other.shape());
        resize_storage(other.size());
        std::copy(other.begin(), other.end(), begin());
        if (other.requires_gradient()) {
            _gradient = new Array(other.shape());
            std::copy(other._gradient->begin(), other._gradient->end(), _gradient->begin());
        }
    }

    void move(Tensor&& other) {
        set_size(other.size());
        set_shape(other.shape());
        resize_storage(other.size());
        std::move(other.begin(), other.end(), begin());
        if (other.requires_gradient()) {
            _gradient = other._gradient;
            other._gradient = nullptr;
        }
    }

    private:
    bool _requires_gradient;
    bool _is_leaf;
    Array* _gradient = nullptr;
    Expression* _expression_view;
};

class Addition : public Expression {
    public:
    Addition(const Tensor& left, const Tensor& right)
    :   _operands{left, right} {}

    void backward(Array& gradient) {
        Array gradient_copy = gradient;
        if (_operands.first.requires_gradient()) {
            _operands.first.backward(gradient);
        }
        if (_operands.second.requires_gradient()) {
            _operands.second.backward(gradient_copy);
        }
    }

    private:
    std::pair<const Tensor&, const Tensor&> _operands;
};

class Multiplication : public Expression {
    public:
    Multiplication(const Tensor& left, const Tensor& right)
    :   _operands{ left, right } {}

    void backward(Array& gradient) {
        Array gradient_copy = gradient;
        if (_operands.first.requires_gradient()) {
            for(auto i = 0; i < gradient.size(); ++i) gradient.data()[i] = gradient_copy.data()[i] * _operands.second.data()[i];
            _operands.first.backward(gradient);
        }
        if (_operands.second.requires_gradient()) {
            for(auto i = 0; i < gradient.size(); ++i) gradient_copy.data()[i] = gradient_copy.data()[i] * _operands.first.data()[i];
            _operands.second.backward(gradient_copy);
        }
    }

    private:
    std::pair<const Tensor&, const Tensor&> _operands;
};

class Buffer {
    public:
    static Buffer& instance() {
        static Buffer instance;
        return instance;
    }

    ~Buffer() { flush(); }

    void flush() {
        for(auto& element : _buffer) delete element;
        _buffer.clear();
    }

    void operator << (Expression* expression) {
        _buffer.push_back(expression);
    }

    private:
    Buffer() = default;
    Buffer(const Buffer&) = delete;
    Buffer(Buffer&&) = delete;
    Buffer& operator=(Buffer&&) = delete;
    Buffer& operator=(const Buffer&) = delete;

    std::vector<Expression*> _buffer;
};

}
///////////////////////////////////

namespace net {

class Tensor {
    public:
    using scalar_type = internal::Array::scalar_type;
    using pointer = internal::Array::pointer;
    using const_pointer = internal::Array::const_pointer;
    using size_type = internal::Tensor::size_type;
    using shape_type = internal::Tensor::shape_type;

    Tensor(shape_type shape, bool requires_gradient = true, bool is_leaf = true) {
        _tensor = std::make_shared<internal::Tensor>(shape, requires_gradient, is_leaf);
    }

    internal::Tensor* operator->() { return _tensor.get(); }
    const internal::Tensor* operator->() const { return _tensor.get(); }

    internal::Tensor& operator*() { return *_tensor; }
    const internal::Tensor& operator*() const { return *_tensor; }

    private:
    std::shared_ptr<internal::Tensor> _tensor;
};

Tensor operator+(const Tensor& first, const Tensor& second) {
    internal::Expression* expression = new internal::Addition(*first, *second);
    internal::Buffer::instance() << expression;
    Tensor result(first->shape(), first->requires_gradient() || second->requires_gradient(), false);
    result->derive_with(expression);
    std::copy(first->begin(), first->end(), result->begin());
    for(auto i = 0; i < result->size(); ++i) result->data()[i] += second->data()[i];
    return result;
}

Tensor operator*(const Tensor& first, const Tensor& second) {
    internal::Expression* expression = new internal::Multiplication(*first, *second);
    internal::Buffer::instance() << expression;
    Tensor result(first->shape(), first->requires_gradient() || second->requires_gradient(), false);
    result->derive_with(expression);
    std::copy(first->begin(), first->end(), result->begin());
    for(auto i = 0; i < result->size(); ++i) result->data()[i] *= second->data()[i];
    return result;
}


}

int main() {
    net::Tensor x({2, 2});
    net::Tensor y({2, 2});
    net::Tensor z({2, 2});
    net::Tensor I({2, 2});

    for(auto i = 0; i < x->size(); ++i) x->data()[i] = 1;
    for(auto i = 0; i < y->size(); ++i) y->data()[i] = -3;
    for(auto i = 0; i < z->size(); ++i) z->data()[i] = 4;
    for(auto i = 0; i < I->size(); ++i) I->data()[i] = 1;

    auto result = x * z + y * z;

    internal::Array gradient({2, 2});
    for(auto i = 0; i < gradient.size(); ++i) gradient.data()[i] = 1;

    result->backward(gradient);

    for(auto i = 0; i < I->size(); ++i) std::cout << result->data()[i] << " ";

    x->print_gradient();
    y->print_gradient();
    z->print_gradient();

    internal::Buffer::instance().flush();

    return 0;
}