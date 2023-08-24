#include <iostream>
#include <vector>
#include <memory>

#include <eigen3/Eigen/Dense>

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

    void copy(const Array* other) {
        size_ = other->size_;
        shape_ = other->shape_;
        storage_ = other->storage_;
    };

    void move(Array* other) {
        size_ = other->size_;
        shape_ = std::move(other->shape_);
        storage_ = std::move(other->storage_);
        other->size_ = 0;
        other->shape_.clear();
        other->storage_.clear();
    };

    Array() = default;
    Array(const Array* other) { copy(other); }
    Array(shape_type shape)
    :   shape_(shape) {
        size_ = 1; for (size_type dimension : shape) size_ *= dimension;
        storage_.resize(size_);
    }

    size_type size() const { return size_; }
    shape_type shape() const { return shape_; }
    size_type rank() const { return shape_.size(); }
    pointer data() { return storage_.data(); }
    const_pointer data() const { return storage_.data(); }

    iterator begin() { return storage_.begin(); }
    iterator end() { return storage_.end(); }
    const_iterator begin() const { return storage_.cbegin(); }
    const_iterator end() const { return storage_.cend(); }
    const_iterator cbegin() const { return storage_.cbegin(); }
    const_iterator cend() const { return storage_.cend(); }

    void add(const Array* other);
    void multiply(const Array* other);

    private:
    size_type size_;
    shape_type shape_;
    storage_type storage_;
};

void Array::add(const Array* other) {
    if(shape_ != other->shape_) throw std::runtime_error("shape mismatch");
    Eigen::Map<Eigen::Array<scalar_type, 1, -1>> this_map(data(), size());
    Eigen::Map<const Eigen::Array<scalar_type, 1, -1>> other_map(other->data(), other->size());
    this_map += other_map;
}

void Array::multiply(const Array* other) {
    if(shape_ != other->shape_) throw std::runtime_error("shape mismatch");
    Eigen::Map<Eigen::Array<scalar_type, 1, -1>> this_map(data(), size());
    Eigen::Map<const Eigen::Array<scalar_type, 1, -1>> other_map(other->data(), other->size());
    this_map *= other_map;
}

struct Expression {
    virtual ~Expression() = default;
    virtual void backward(Array* gradient) = 0;
};


class Tensor : public Array {
    public:
    Tensor(const Tensor* other) { copy(other); }
    Tensor(const Tensor& other) { copy(&other); }
    Tensor(Tensor&& other) { move(&other); }
    Tensor& operator=(const Tensor& other) { if (this != &other) copy(&other);  return *this; }
    Tensor& operator=(Tensor&& other) { if (this != &other) move(&other); return *this; }
    ~Tensor() override { if (requires_gradient_) delete gradient_; }
    Tensor(Array&& other) { Array::move(&other); }

    Tensor(shape_type shape) : Array(shape) {
        if (requires_gradient_) { 
            gradient_ = new Array(shape);
        }
    }
    
    void backward(Array* gradient) const {
        if (is_leaf_) { gradient_->add(gradient); } 
        else { expression_view_->backward(gradient); }
    }

    void copy(const Tensor* other) {
        Array::copy(other);
        if (requires_gradient_) { delete gradient_; }
        if (other->requires_gradient_) {
            gradient_ = new Array(other);
        }
        requires_gradient_ = other->requires_gradient_;
        is_leaf_ = other->is_leaf_;
    }

    void move(Tensor* other) {
        Array::move(other);
        if (requires_gradient_) { delete gradient_; }
        if (other->requires_gradient_) {
            gradient_ = other->gradient_;
            other->gradient_ = nullptr;    
        } 
        requires_gradient_ = other->requires_gradient_;
        is_leaf_ = other->is_leaf_;
    }

    Array* gradient() const { return gradient_; }
    bool is_leaf() const { return is_leaf_; }
    void is_leaf(bool status) { is_leaf_ = status; }
    void derive_with(Expression* expression) { expression_view_ = expression; }
    bool requires_gradient() const { return requires_gradient_; }
    void requires_gradient(bool status) {        
        if (requires_gradient_ == false && status == true) {
            requires_gradient_ = true;
            gradient_ = new Array(shape());
        }

        if (requires_gradient_ == true && status == false ) {
            requires_gradient_ = false;
            delete gradient_;
            gradient_ = nullptr;
        }
    }


    void print_gradient() {
        if(requires_gradient_) {
            for(auto e : *gradient_) std::cout << e;
        }
    }

    private:
    bool requires_gradient_ = false;
    bool is_leaf_ = false;
    Array* gradient_ = nullptr;
    Expression* expression_view_ = nullptr;
};

class BinaryExpression : public Expression {
    public:
    ~BinaryExpression() override = default;

    BinaryExpression(const Tensor* first, const Tensor* second)
    :   operands{ first, second }
    ,   gradient_requirement(first->requires_gradient() || second->requires_gradient())
    {}

    virtual Tensor perform() const = 0;

    bool gradient_requirement;
    std::pair<const Tensor*, const Tensor*> operands;
};

class Addition : public BinaryExpression {
    public:
    using BinaryExpression::BinaryExpression;
    ~Addition() final = default;
    Tensor perform() const final;
    void backward(Array* gradient) final;
};

Tensor Addition::perform() const {
    Tensor result(operands.first);
    result.add(operands.second);
    result.requires_gradient(this->gradient_requirement);
    result.is_leaf(false);
    return result;
}

void Addition::backward(Array* gradient) {
    Array* gradient_copy = new Array(gradient);
    if (operands.first->requires_gradient()) {
        operands.first->backward(gradient);
    }
    if (operands.second->requires_gradient()) {
        operands.second->backward(gradient_copy);
    }
    delete gradient_copy;
}

class Multiplication : public BinaryExpression {
    public:
    using BinaryExpression::BinaryExpression;
    ~Multiplication() final = default;
    Tensor perform() const final;
    void backward(Array* gradient) final;
};

Tensor Multiplication::perform() const  {
    Tensor result(this->operands.first);
    result.multiply(this->operands.second);
    result.requires_gradient(this->gradient_requirement);
    result.is_leaf(false);
    return result;
}

void Multiplication::backward(Array* gradient) {
    Array* gradient_copy = new Array(gradient);
    if (operands.first->requires_gradient()) {
        gradient->multiply(this->operands.second);
        operands.first->backward(gradient);
    }
    if (operands.second->requires_gradient()) {
        gradient_copy->multiply(this->operands.first);
        operands.second->backward(gradient_copy);
    }
    delete gradient_copy;
}


class MatrixMultiplication : public BinaryExpression {
    public:
    using scalar_type = Tensor::scalar_type;
    using size_type = Tensor::size_type;

    MatrixMultiplication(const Tensor* first, const Tensor* second);
    ~MatrixMultiplication() final = default;

    size_type rows;
    size_type columns;
    size_type inner_dimension;

    Tensor perform() const final ;
    void backward(Array* gradient) final ;
};

MatrixMultiplication::MatrixMultiplication(const Tensor* first, const Tensor* second)
:   BinaryExpression(first, second)
,   rows(operands.first->shape().front())
,   columns(operands.second->shape().back())
,   inner_dimension(operands.first->shape().back()) {
    if (first->rank() != 2 || second->rank() != 2) throw std::runtime_error("rank mismatch");
    if (first->shape().back() != second->shape().front()) throw std::runtime_error("shape mismatch");
}

Tensor MatrixMultiplication::perform() const {
    Tensor result({rows, columns});

    Eigen::Map<Eigen::Matrix<scalar_type, -1, -1, 1>> result_map(result.data(), rows, columns);
    Eigen::Map<const Eigen::Matrix<scalar_type, -1, -1, 1>> first_map(operands.first->data(), rows, inner_dimension);
    Eigen::Map<const Eigen::Matrix<scalar_type, -1, -1, 0>> second_map(operands.second->data(), inner_dimension, columns);
    
    result_map = first_map * second_map;
    result.requires_gradient(this->gradient_requirement);
    result.is_leaf(false);
    return result;
}

void MatrixMultiplication::backward(Array* gradient) {
    Eigen::Map<const Eigen::Matrix<scalar_type, -1, -1, 1>> row_gradient_map(gradient->data(), rows, columns);
    if (operands.first->requires_gradient()) {
        Array* first_gradient = new Array({rows, inner_dimension});
        Eigen::Map<const Eigen::Matrix<scalar_type, -1, -1, 1>> second_map(operands.second->data(), inner_dimension, columns);
        Eigen::Map<Eigen::Matrix<scalar_type, -1, -1, 1>> first_gradient_map(first_gradient->data(), rows, inner_dimension);
        first_gradient_map = row_gradient_map * second_map.transpose();
        first_gradient_map.eval();
        operands.first->backward(first_gradient);
        delete first_gradient;
    }
    
    Eigen::Map<const Eigen::Matrix<scalar_type, -1, -1, 0>> column_gradient_map(gradient->data(), rows, columns);
    if (operands.second->requires_gradient()) {
        Array* second_gradient = new Array({inner_dimension, columns});
        Eigen::Map<const Eigen::Matrix<scalar_type, -1, -1, 0>> first_map(operands.first->data(), rows, inner_dimension);
        Eigen::Map<Eigen::Matrix<scalar_type, -1, -1, 0>> second_gradient_map(second_gradient->data(), inner_dimension, columns);
        second_gradient_map = first_map.transpose() * column_gradient_map;
        second_gradient_map.eval();
        operands.second->backward(second_gradient);
        delete second_gradient;
    }
}

class Linear : public Expression {
    public:
    using scalar_type = Tensor::scalar_type;
    using size_type = Tensor::size_type;
    using shape_type = Tensor::shape_type;

    Linear(const Tensor* input, const Tensor* weights, const Tensor* bias);
    ~Linear() final = default;

    Tensor perform() const;
    void backward(Array* gradient) final;

    bool gradient_requirement;

    const Tensor* input;
    const Tensor* weight;
    const Tensor* bias;

    size_type rows;
    size_type columns;
    size_type inner_dimension;
};

Linear::Linear(const Tensor* input, const Tensor* weight, const Tensor* bias)
:   input(input)
,   weight(weight)
,   bias(bias)
,   gradient_requirement(weight->requires_gradient() || bias->requires_gradient())
,   rows(input->shape().front())
,   columns(weight->shape().back())
,   inner_dimension(input->shape().back()) {
    if (input->rank() != 2 || weight->rank() != 2) throw std::runtime_error("rank mismatch");
    if (input->shape().back() != weight->shape().front()) throw std::runtime_error("shape mismatch");
}

Tensor Linear::perform() const {
    Tensor result({rows, columns});

    Eigen::Map<Eigen::Matrix<scalar_type, -1, -1, 1>> result_map(result.data(), rows, columns);
    Eigen::Map<const Eigen::Matrix<scalar_type, -1, -1, 1>> input_map(input->data(), rows, inner_dimension);
    Eigen::Map<const Eigen::Matrix<scalar_type, -1, -1, 0>> weight_map(weight->data(), inner_dimension, columns);
    Eigen::Map<const Eigen::Matrix<scalar_type, 1, -1>> bias_map(bias->data(), columns);

    result_map = (input_map * weight_map).rowwise() + bias_map;
    result_map.eval();

    result.requires_gradient(this->gradient_requirement);
    result.is_leaf(false);
    return result;
}

void Linear::backward(Array* gradient) {
    Eigen::Map<const Eigen::Matrix<scalar_type, -1, -1, 1>> row_gradient_map(gradient->data(), rows, columns);

    if (input->requires_gradient()) {
        Array* input_gradient = new Array({rows, inner_dimension});
        Eigen::Map<const Eigen::Matrix<scalar_type, -1, -1, 1>> weight_map(weight->data(), inner_dimension, columns);
        Eigen::Map<Eigen::Matrix<scalar_type, -1, -1, 1>> input_gradient_map(input_gradient->data(), rows, inner_dimension);
        input_gradient_map = row_gradient_map * weight_map.transpose();
        input_gradient_map.eval();
        input->backward(input_gradient);
        delete input_gradient;
    }
    
    Eigen::Map<const Eigen::Matrix<scalar_type, -1, -1, 0>> column_gradient_map(gradient->data(), rows, columns);

    if (weight->requires_gradient()) {
        Array* weight_gradient = new Array({inner_dimension, columns});
        Eigen::Map<const Eigen::Matrix<scalar_type, -1, -1, 0>> input_map(input->data(), rows, inner_dimension);
        Eigen::Map<Eigen::Matrix<scalar_type, -1, -1, 0>> weight_gradient_map(weight_gradient->data(), inner_dimension, columns);
        weight_gradient_map = input_map.transpose() * column_gradient_map;
        weight_gradient_map.eval();
        weight->backward(weight_gradient);
        delete weight_gradient;
    }
    
    if (bias->requires_gradient()) {
        Array* bias_gradient = new Array({columns});
        Eigen::Map<Eigen::Matrix<scalar_type, 1, -1>> bias_gradient_map(bias_gradient->data(), columns);
        bias_gradient_map = row_gradient_map.rowwise().sum();
        bias->backward(bias_gradient);
        delete bias_gradient;
    }
}

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

} // namespace internal

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
    Tensor(shape_type shape, bool requires_gradient = true, bool is_leaf = true);
    internal::Tensor* internal() const;

    iterator begin();
    iterator end();
    const_iterator begin() const;
    const_iterator end() const;
    const_iterator cbegin() const;
    const_iterator cend() const;

    private:
    std::shared_ptr<internal::Tensor> _tensor;
};


Tensor::Tensor(std::shared_ptr<internal::Tensor> tensor)
:   _tensor(tensor) {}

Tensor::Tensor(shape_type shape, bool gradient_requirement, bool node_status ) {
    _tensor = std::make_shared<internal::Tensor>(shape);
    _tensor->requires_gradient(gradient_requirement);
    _tensor->is_leaf(node_status);
}

internal::Tensor* Tensor::internal() const {return _tensor.get(); }

Tensor::iterator Tensor::begin() { return _tensor->begin(); }
Tensor::iterator Tensor::end() { return _tensor->end(); }
Tensor::const_iterator Tensor::begin() const { return _tensor->begin(); }
Tensor::const_iterator Tensor::end() const { return _tensor->end(); }
Tensor::const_iterator Tensor::cbegin() const { return _tensor->cbegin(); }
Tensor::const_iterator Tensor::cend() const { return _tensor->cend(); }

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


} // namespace net


int main() {
    internal::Tensor X({2, 3}); for(auto& e : X) e = 1;
    internal::Tensor W({3, 2}); for(auto& e : W) e = 2;
    internal::Tensor b({2}); for(auto& e : b) e = 3;

    X.requires_gradient(true);
    W.requires_gradient(true);
    b.requires_gradient(true);

    X.is_leaf(true);
    W.is_leaf(true);
    b.is_leaf(true);
    
    internal::Array I({2, 2}); for(auto& e : I) e = 1;

    internal::Linear linear(&X, &W, &b);

    internal::Tensor Y({2, 2});
    Y.requires_gradient(true);
    Y.is_leaf(false);

    Y = linear.perform();
    linear.backward(&I);

    for (auto& e : Y) std::cout << e << " ";

    X.print_gradient();
    W.print_gradient();
    b.print_gradient();
    return 0;
}

