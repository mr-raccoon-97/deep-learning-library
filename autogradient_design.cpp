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
    bool requires_gradient;
    bool is_leaf;

    Tensor(const Tensor& other) : requires_gradient(other.requires_gradient) , is_leaf(other.is_leaf) { copy(other); }
    Tensor(Tensor&& other) : requires_gradient(other.requires_gradient) , is_leaf(other.is_leaf)  { move(std::move(other)); }
    Tensor& operator=(const Tensor& other) { if (this != &other) copy(other);  return *this; }
    Tensor& operator=(Tensor&& other) { if (this != &other) move(std::move(other)); return *this; }
    ~Tensor() { if (requires_gradient) delete _gradient; }

    Tensor(shape_type shape, bool gradient_requirement, bool node_status)
    :   Array(shape)
    ,   requires_gradient(gradient_requirement)
    ,   is_leaf(node_status) {
        if (requires_gradient) { 
            _gradient = new Array(shape);
        }
    }
    
    void derive_with(Expression* expression) { _expression_view = expression; }

    void backward(Array& gradient) const {
        if (is_leaf) { _gradient->add(gradient); } 
        else { _expression_view->backward(gradient); }
    }

    void print_gradient() {
        if (requires_gradient) {
            for(auto i = 0; i < _gradient->size(); ++i) std::cout << _gradient->data()[i] << " ";
        }
    }

    void copy(const Tensor& other) {
        set_size(other.size());
        set_shape(other.shape());
        resize_storage(other.size());
        std::copy(other.begin(), other.end(), begin());
        if (other.requires_gradient) {
            _gradient = new Array(other.shape());
            std::copy(other._gradient->begin(), other._gradient->end(), _gradient->begin());
        }
    }

    void move(Tensor&& other) {
        set_size(other.size());
        set_shape(other.shape());
        resize_storage(other.size());
        std::move(other.begin(), other.end(), begin());
        if (other.requires_gradient) {
            _gradient = other._gradient;
            other._gradient = nullptr;
        }
    }
    private:
    Array* _gradient = nullptr;
    Expression* _expression_view = nullptr;
};

class BinaryExpression : public Expression {
    public:
    ~BinaryExpression() override = default;
    BinaryExpression(const Tensor& first, const Tensor& second)
    :   operands{ first, second }
    ,   gradient_requirement(first.requires_gradient || second.requires_gradient)
    {
        if (first.shape() != second.shape()) throw std::runtime_error("shape mismatch");
    }
    Tensor::shape_type shape() const { return operands.first.shape(); }

    virtual Tensor perform() const = 0;

    protected:
    std::pair<const Tensor&, const Tensor&> operands;
    bool gradient_requirement;
};

class Addition : public BinaryExpression {
    public:
    using BinaryExpression::BinaryExpression;
    using BinaryExpression::backward;
    ~Addition() final = default;

    Tensor perform() const final {
        Tensor result(this->shape(), this->gradient_requirement, false);
        result.copy(operands.first);
        result.add(operands.second);
        return result;
    }

    void backward(Array& gradient) final {
        Array gradient_copy = gradient;
        if (operands.first.requires_gradient) {
            operands.first.backward(gradient);
        }
        if (operands.second.requires_gradient) {
            operands.second.backward(gradient_copy);
        }
    }
};

class Multiplication : public BinaryExpression {
    public:
    using BinaryExpression::BinaryExpression;
    using BinaryExpression::backward;
    ~Multiplication() final = default;

    Tensor perform() const final {
        Tensor result(this->shape(), this->gradient_requirement, false);
        result.copy(operands.first);
        result.multiply(operands.second);
        return result;
    }

    void backward(Array& gradient) {
        Array gradient_copy = gradient;
        if (operands.first.requires_gradient) {
            gradient.multiply(operands.second);
            operands.first.backward(gradient);
        }
        if (operands.second.requires_gradient) {
            gradient_copy.multiply(operands.first);
            operands.second.backward(gradient_copy);
        }
    }
};

class Buffer {
    public:
    static Buffer& instance() { static Buffer instance; return instance; }
    ~Buffer() { flush(); }
    void flush() { for(auto& element : _buffer) delete element; _buffer.clear(); }
    void operator << (Expression* expression) { _buffer.push_back(expression); }

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
    using shape_type = internal::Array::shape_type;
    using size_type = internal::Array::size_type;

    Tensor(std::shared_ptr<internal::Tensor> tensor)
    :   _tensor(tensor) {}

    Tensor(shape_type shape, bool requires_gradient = true, bool is_leaf = true) {
        _tensor = std::make_shared<internal::Tensor>(shape, requires_gradient, is_leaf);
    }

    internal::Tensor* operator -> () { return _tensor.get(); }
    internal::Tensor& operator * () const { return *_tensor; }


    private:
    std::shared_ptr<internal::Tensor> _tensor;
};


Tensor operator + (const Tensor& left, const Tensor& right) {
    internal::BinaryExpression* expression = new internal::Addition(*left, *right);
    std::shared_ptr<internal::Tensor> internal_result = std::make_shared<internal::Tensor>(expression->perform());
    internal_result->derive_with(expression);
    Tensor result(std::move(internal_result));
    internal::Buffer::instance() << expression;
    return result;
}

Tensor operator * (const Tensor& left, const Tensor& right) {
    internal::BinaryExpression* expression = new internal::Multiplication(*left, *right);
    std::shared_ptr<internal::Tensor> internal_result = std::make_shared<internal::Tensor>(expression->perform());
    internal_result->derive_with(expression);
    Tensor result(std::move(internal_result));
    internal::Buffer::instance() << expression;
    return result;
}

}

int main() {
    net::Tensor x({2, 2});
    net::Tensor y({2, 2});
    net::Tensor z({2, 2});


    for(auto i = 0; i < x->size(); ++i) x->data()[i] = 1;
    for(auto i = 0; i < y->size(); ++i) y->data()[i] = -3;
    for(auto i = 0; i < z->size(); ++i) z->data()[i] = 4;

    auto result = x * z + y * z;

    internal::Array gradient({2, 2});
    for(auto i = 0; i < gradient.size(); ++i) gradient.data()[i] = 1;

    result->backward(gradient);
    
    x->print_gradient();
    y->print_gradient();
    z->print_gradient();

    internal::Buffer::instance().flush();

    return 0;
}