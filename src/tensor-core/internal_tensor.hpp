#ifndef INTERNAL_TENSOR_HPP
#define INTERNAL_TENSOR_HPP

#include <iostream>
#include <vector>
#include <memory>

#include "internal_array.hpp"
#include "internal_expression.hpp"

namespace internal {

class Tensor : public Array {
    public:

    Tensor(shape_type shape) : Array(shape) {}
    Tensor(const Tensor* other) { copy(other); }
    Tensor(const Tensor& other) { copy(&other); }
    Tensor(Tensor&& other) { move(&other); }
    Tensor& operator=(const Tensor& other) { if (this != &other) copy(&other);  return *this; }
    Tensor& operator=(Tensor&& other) { if (this != &other) move(&other); return *this; }
    ~Tensor() override { if (requires_gradient_) delete gradient_; }
    Tensor(Array&& other) { Array::move(&other); }
    
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

} // namespace internal

#endif // INTERNAL_TENSOR_HPP