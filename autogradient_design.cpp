#include <iostream>
#include <vector>

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
    bool gradient_requirement;
    std::pair<const Tensor*, const Tensor*> operands;

    ~BinaryExpression() override = default;

    BinaryExpression(const Tensor* first, const Tensor* second)
    :   operands{ first, second }
    ,   gradient_requirement(first->requires_gradient() || second->requires_gradient())
    {}

    virtual Tensor perform() const = 0;
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

    size_type rows;
    size_type columns;
    size_type inner_dimension;

    MatrixMultiplication(const Tensor* first, const Tensor* second);
    ~MatrixMultiplication() final = default;

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


} // namespace internal

int main() {
    internal::Tensor x({2,3}); for(auto& e : x) e = 1;
    x.requires_gradient(true);
    x.is_leaf(true);

    internal::Tensor y({3,2}); for(auto& e : y) e = 1;
    y.requires_gradient(true);
    y.is_leaf(true);

    internal::MatrixMultiplication m(&x,&y);
    internal::Tensor z = m.perform();
    z.requires_gradient(true);
    z.is_leaf(false);

    internal::Array I({2,2}); for(auto& e : I) e = 1;

    m.backward(&I);

    x.print_gradient();
    y.print_gradient();

    return 0;
}

