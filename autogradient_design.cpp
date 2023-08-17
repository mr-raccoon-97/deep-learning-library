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

    virtual ~Array() = default;

    virtual void copy(const Array* other) {
        _size = other->_size;
        _shape = other->_shape;
        _storage = other->_storage;
    };

    virtual void move(Array* other) {
        _size = other->_size;
        _shape = std::move(other->_shape);
        _storage = std::move(other->_storage);
        other->_size = 0;
        other->_shape.clear();
        other->_storage.clear();
    };

    Array() = default;
    Array(const Array* other) { copy(other); }
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

    Array& add (const Array& other); //
    Array& multiply (const Array& other); //

    void add(const Array* other);
    void multiply(const Array* other);


    protected:
    void set_size(size_type size) { _size = size; }
    void set_shape(shape_type shape) { _shape = shape; }
    void resize_storage(size_type size) { _storage.resize(size); }

    private:
    size_type _size;
    shape_type _shape;
    storage_type _storage;
};

void Array::add(const Array* other) {
    if(_shape != other->_shape) throw std::runtime_error("shape mismatch");
    for(size_type i = 0; i < _size; ++i) _storage[i] += other->_storage[i];
}


void Array::multiply(const Array* other) {
    if(_shape != other->_shape) throw std::runtime_error("shape mismatch");
    for(size_type i = 0; i < _size; ++i) _storage[i] *= other->_storage[i];
}

struct Expression {
    virtual ~Expression() = default;
    virtual void backward(Array* gradient) = 0;
};

class Tensor : public Array {
    public:
    bool requires_gradient;
    bool is_leaf;


    Tensor(const Tensor* other) : requires_gradient(other->requires_gradient) , is_leaf(other->is_leaf) { copy(other); }
    Tensor(const Tensor& other) : requires_gradient(other.requires_gradient) , is_leaf(other.is_leaf) { copy(&other); }
    Tensor(Tensor&& other) : requires_gradient(other.requires_gradient) , is_leaf(other.is_leaf)  { move(&other); }
    Tensor& operator=(const Tensor& other) { if (this != &other) copy(&other);  return *this; }
    Tensor& operator=(Tensor&& other) { if (this != &other) move(&other); return *this; }
    ~Tensor() override { if (requires_gradient) delete _gradient; }

    Tensor(shape_type shape, bool gradient_requirement, bool node_status)
    :   Array(shape)
    ,   requires_gradient(gradient_requirement)
    ,   is_leaf(node_status) {
        if (requires_gradient) { 
            _gradient = new Array(shape);
        }
    }
    
    void derive_with(Expression* expression) { _expression_view = expression; }

    void backward(Array* gradient) const {
        if (is_leaf) { _gradient->add(gradient); } 
        else { _expression_view->backward(gradient); }
    }

    void print_gradient() {
        if (requires_gradient) {
            for(auto i = 0; i < _gradient->size(); ++i) std::cout << _gradient->data()[i] << " ";
        }
    }

    void copy(const Array* other) final {
        Array::copy(other);
        requires_gradient = false;
        is_leaf = false;
    }

    void move(Array* other) final {
        Array::move(other);
        requires_gradient = false;
        is_leaf = false;
    }

    void copy(const Tensor* other) {
        Array::copy(other);
        requires_gradient = other->requires_gradient;
        if (requires_gradient) {
            _gradient = new Array(other);
        }
    }

    void move(Tensor* other) {
        Array::move(other);
        requires_gradient = other->requires_gradient;
        if (requires_gradient) {
            _gradient = other->_gradient;
            other->_gradient = nullptr;
        }
    }

    void copy(const Tensor& other) { copy(&other); }
    void move(Tensor&& other) { move(&other); }

    private:
    Array* _gradient = nullptr;
    Expression* _expression_view = nullptr;
};

class BinaryExpression : public Expression {
    public:
    ~BinaryExpression() override = default;

    BinaryExpression(const Tensor* first, const Tensor* second)
    :   operands{ first, second }
    ,   gradient_requirement(first->requires_gradient || second->requires_gradient)
    {
        if (first->shape() != second->shape()) throw std::runtime_error("shape mismatch");
    }

    Tensor::shape_type shape() const { return operands.first->shape(); }

    virtual Tensor perform() const = 0;

    protected:
    std::pair<const Tensor*, const Tensor*> operands;
    bool gradient_requirement;
};

class Addition : public BinaryExpression {
    public:
    using BinaryExpression::BinaryExpression;
    using BinaryExpression::backward;
    ~Addition() final = default;

    Tensor perform() const final {
        Tensor result(operands.first);
        result.add(operands.second);
        result.requires_gradient = this->gradient_requirement;
        result.is_leaf = false;
        return result;
    }

    void backward(Array* gradient) final {
        Array* gradient_copy = new Array(gradient);
        if (operands.first->requires_gradient) {
            operands.first->backward(gradient);
        }
        if (operands.second->requires_gradient) {
            operands.second->backward(gradient_copy);
        }
        delete gradient_copy;
    }
};

class Multiplication : public BinaryExpression {
    public:
    using BinaryExpression::BinaryExpression;
    using BinaryExpression::backward;
    ~Multiplication() final = default;

    Tensor perform() const final {
        Tensor result(operands.first);
        result.multiply(operands.second);
        result.requires_gradient = this->gradient_requirement;
        result.is_leaf = false;
        return result;
    }


    void backward(Array* gradient) {
        Array* gradient_copy = new Array(gradient);
        if (operands.first->requires_gradient) {
            gradient->multiply(operands.second);
            operands.first->backward(gradient);
        }
        if (operands.second->requires_gradient) {
            gradient_copy->multiply(operands.first);
            operands.second->backward(gradient_copy);
        }
        delete gradient_copy;
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

    void backward(const net::Tensor& gradient) const { _tensor->backward(gradient.internal()); }

    internal::Tensor* internal() const {return _tensor.get(); }

    auto begin() { return _tensor->begin(); }
    auto end() { return _tensor->end(); }
    auto cbegin() const { return _tensor->cbegin(); }
    auto cend() const { return _tensor->cend(); }

    private:
    std::shared_ptr<internal::Tensor> _tensor;
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