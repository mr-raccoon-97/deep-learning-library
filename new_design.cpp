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

    virtual void backward(Array* gradient) {
    // Add gradient to the current gradient.
    };

    virtual Tensor* forward() {
        return this;
    }

    private:
    bool requires_gradient_ = false;
    bool is_leaf_ = false;
    Array* gradient_ = nullptr;
};

class Operation : public Tensor {
    public:
    Operation(Tensor* first, Tensor* second) 
    :   operands_{ first, second } {
        requires_gradient(first->requires_gradient() || second->requires_gradient());
    }

    Tensor* first_operand() const { return operands_.first; }
    Tensor* second_operand() const { return operands_.second; }
    
    private:
    std::pair<Tensor*, Tensor*> operands_;
};

class Addition : public Operation {
    public:
    Addition(Tensor* first, Tensor* second) : Operation(first, second) {
        reshape(first->shape());
    }

    Tensor* forward() final;
    void backward(Array* gradient) final;
};

Tensor* Addition::forward() {
    Tensor* addend = first_operand()->forward();
    Tensor* augend = second_operand()->forward();

    Eigen::Map<Eigen::Array<scalar_type, 1, -1>> this_map(
        this->data(),
        this->size() );

    Eigen::Map<const Eigen::Array<scalar_type, 1, -1>> addend_map(
        addend->data(),
        addend->size() );
        
    Eigen::Map<const Eigen::Array<scalar_type, 1, -1>> augend_map(
        augend->data(),
        augend->size() );

    this_map = addend_map + augend_map;
    return this;
}

void Addition::backward(Array* gradient) {
    if (first_operand()->requires_gradient()) {
        if (second_operand()->requires_gradient()) {
            Array* gradient_copy = new Array(gradient);
            first_operand()->backward(gradient_copy);
            delete gradient_copy;
        }
        
        else {
            first_operand()->backward(gradient);
        }
    }

    if (second_operand()->requires_gradient()) {
        second_operand()->backward(gradient);
    }
}

class Multiplication : public Operation {
    public:
    Multiplication(Tensor* first, Tensor* second) : Operation(first, second) {
        reshape(first->shape());
    }

    Tensor* forward() final;
    void backward(Array* gradient) final;
};

Tensor* Multiplication::forward() {
    Tensor* multiplicand = first_operand()->forward();
    Tensor* multiplier = second_operand()->forward();

    Eigen::Map<Eigen::Array<scalar_type, 1, -1>> this_map(
        this->data(),
        this->size() );

    Eigen::Map<const Eigen::Array<scalar_type, 1, -1>> multiplicand_map(
        multiplicand->data(),
        multiplicand->size() );
        
    Eigen::Map<const Eigen::Array<scalar_type, 1, -1>> multiplier_map(
        multiplier->data(),
        multiplier->size() );

    this_map = multiplicand_map * multiplier_map;
    return this;
}

void Multiplication::backward(Array* gradient) {
    Eigen::Map<Eigen::Array<scalar_type, 1, -1>> gradient_map(
        gradient->data(),
        gradient->size()
    );

    if (first_operand()->requires_gradient()) {
        Eigen::Map<const Eigen::Array<scalar_type, 1, -1>> second_operand_map(
            second_operand()->data(),
            second_operand()->size()
        );
        
        if (second_operand()->requires_gradient()) {
            Array* gradient_copy = new Array(gradient);
            Eigen::Map<Eigen::Array<scalar_type, 1, -1>> gradient_copy_map(
                gradient_copy->data(),
                gradient_copy->size()
            );
            gradient_copy_map *= second_operand_map;
            first_operand()->backward(gradient_copy);
            delete gradient_copy;
        }
        
        else {
            gradient_map *= second_operand_map;
            first_operand()->backward(gradient);
        }
    }

    if (second_operand()->requires_gradient()) {
        Eigen::Map<const Eigen::Array<scalar_type, 1, -1>> first_operand_map(
            first_operand()->data(),
            first_operand()->size()
        );

        gradient_map *= first_operand_map;
        second_operand()->backward(gradient);
    }
}

} // namespace internal

int main() {
    internal::Tensor x({2,2}); for(auto& element : x) element = 1; x.requires_gradient(true); x.is_leaf(true);
    internal::Tensor y({2,2}); for(auto& element : y) element = -3; y.requires_gradient(true); y.is_leaf(true);
    internal::Tensor z({2,2}); for(auto& element : z) element = 4; z.requires_gradient(true); z.is_leaf(true);

    internal::Tensor* a = new internal::Multiplication(&x, &z);
    internal::Tensor* b = new internal::Multiplication(&y, &z);
    internal::Tensor* c = new internal::Addition(a, b);

    internal::Array* gradient = new internal::Array({2,2}); for(auto& element : *gradient) element = 1;

    internal::Tensor* result = c->forward();
    result->backward(gradient);

    for (auto e : *result) std::cout << e;

    delete a;
    delete b;
    delete c;
    delete gradient;
}