#include <iostream>
#include <memory>
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

    Array() = default;
    Array(const Array* other) { copy(other); }
    Array(shape_type shape) { reshape(shape); }

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

    void reshape(const shape_type& shape) {
        shape_ = shape;
        size_ = 1; for (size_type dimension : shape) size_ *= dimension;
        storage_.resize(size_);
    }

    private:
    size_type size_;
    shape_type shape_;
    storage_type storage_;
};

class Tensor : public Array {
    public:
    Tensor() = default;
    Tensor(shape_type shape) : Array(shape) {}
    Tensor(const Tensor* other) { copy(other); }
    Tensor(const Tensor& other) { copy(&other); }
    Tensor(Tensor&& other) { move(&other); }
    Tensor& operator=(const Tensor& other) { if (this != &other) copy(&other);  return *this; }
    Tensor& operator=(Tensor&& other) { if (this != &other) move(&other); return *this; }
    ~Tensor() override { if (requires_gradient_) delete gradient_; }
    Tensor(Array&& other) { Array::move(&other); }

    void copy(const Tensor* other) {
        Array::copy(other);
        if (requires_gradient_) { delete gradient_; }
        if (other->requires_gradient_) {
            gradient_ = new Array(other);
        }
        requires_gradient_ = other->requires_gradient_;
    }

    void move(Tensor* other) {
        Array::move(other);
        if (requires_gradient_) { delete gradient_; }
        if (other->requires_gradient_) {
            gradient_ = other->gradient_;
            other->gradient_ = nullptr;    
        } 
        requires_gradient_ = other->requires_gradient_;
    }

    Array* gradient() const { return gradient_; }
    
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

    bool is_leaf() const { return is_leaf_; }
    void is_leaf(bool status) { is_leaf_ = status; }

    private:
    bool is_leaf_ = false;
    bool requires_gradient_ = false;
    Array* gradient_ = nullptr;
};

class Expression {
    public:
    using scalar_type = Tensor::scalar_type;
    using size_type = Tensor::size_type;
    using shape_type = Tensor::shape_type;

    virtual ~Expression() = default;
    virtual void backward(Array* gradient) const = 0;
    virtual void forward() = 0;
    virtual shape_type shape() const = 0;
    virtual bool gradient_requirement() const = 0;
};

class Operation : public Expression {
    public:
    Operation(Tensor* first, Tensor* second) {
        first_operand_ = first;
        second_operand_ = second;
    }

    Tensor* first_operand() const { return first_operand_; }
    Tensor* second_operand() const { return second_operand_; }

    private:
    Tensor* first_operand_;
    Tensor* second_operand_;
    Tensor* result_;
};

class Graph {
    public:
    static Graph& instance() {
        static Graph instance;
        return instance;
    }

    void add_expression(Expression* expression) { expressions_.push_back(expression); }
    void add_tensor(Tensor* tensor) { tensors_.push_back(tensor); }



    private:
    std::vector<Expression*> expressions_;
    std::vector<Tensor*> tensors_;
};

} // namespace internal

class Tensor {

    private:
    std::shared_ptr<internal::Tensor> tensor_;
};


Tensor operator + (const Tensor& first, const Tensor& second) {
    internal::Expression* expression = new internal::Addition(first.internal(), second.internal());
    Tensor(expression->shape(), expression->gradient_requirement(), false);
}


int main() {
}