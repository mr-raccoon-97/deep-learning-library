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
    ~Tensor() override { if (is_leaf_) delete gradient_; }
    Tensor(Array&& other) { Array::move(&other); }

    void copy(const Tensor* other) {
        Array::copy(other);
        requires_gradient_ = other->requires_gradient_;

        if (requires_gradient_ ) {

            if (other->is_leaf_ && is_leaf_) {
                if (!gradient_) gradient_ = new Array(other->gradient_);
                else gradient_->copy(other->gradient_);
            }

            else if (other->is_leaf_ && !is_leaf_) {
                gradient_ = new Array(other->gradient_);
            }

            else {
                if (is_leaf_) delete gradient_;
                gradient_ = other->gradient_;
            }
        }
        
        else {
            if (is_leaf_) delete gradient_;
            gradient_ = nullptr;
        }

        is_leaf_ = other->is_leaf_;
    }

    void move(Tensor* other) {
        Array::move(other);
        if (is_leaf_) delete gradient_;
        is_leaf_ = other->is_leaf_;
        requires_gradient_ = other->requires_gradient_;
        gradient_ = other->gradient_;
        other->gradient_ = nullptr;
    }

    Array* gradient() const { return gradient_; }
    
    bool requires_gradient() const { return requires_gradient_; }

    void requires_gradient(bool status) {        

        if (requires_gradient_ == false && status == true) {
            requires_gradient_ = true;
            if (is_leaf_) gradient_ = new Array(shape());
        }

        if (requires_gradient_ == true && status == false ) {
            requires_gradient_ = false;
            if (is_leaf_) delete gradient_;
            gradient_ = nullptr;
        }
    }

    bool is_leaf() const { return is_leaf_; }
    void is_leaf(bool status) { is_leaf_ = status; }

    virtual void backward(Array* gradient) const {
    }

    virtual Tensor* forward() { return this; }

    private:
    bool is_leaf_ = true;
    bool requires_gradient_ = false;
    Array* gradient_ = nullptr;
};

}


int main() {
    internal::Tensor* t = new internal::Tensor({2, 3});
    for (auto& x : *t) x = 1;

    internal::Tensor* t2 = new internal::Tensor();
    t2->copy(t);

    for(auto x : t2->shape()) std::cout << x << std::endl;
    return 0;    
}