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

    Tensor(const Tensor* other) : requires_gradient_(other->requires_gradient_) , is_leaf_(other->is_leaf_) { copy(other); }
    Tensor(const Tensor& other) : requires_gradient_(other.requires_gradient_) , is_leaf_(other.is_leaf_) { copy(&other); }
    Tensor(Tensor&& other) : requires_gradient_(other.requires_gradient_) , is_leaf_(other.is_leaf_)  { move(&other); }
    Tensor& operator=(const Tensor& other) { if (this != &other) copy(&other);  return *this; }
    Tensor& operator=(Tensor&& other) { if (this != &other) move(&other); return *this; }
    ~Tensor() override { if (requires_gradient_) delete gradient_; }

    Tensor(shape_type shape, bool gradient_requirement, bool node_status)
    :   Array(shape)
    ,   requires_gradient_(gradient_requirement)
    ,   is_leaf_(node_status) {
        if (requires_gradient_) { 
            gradient_ = new Array(shape);
        }
    }
    
    void derive_with(Expression* expression) { expression_view_ = expression; }

    void backward(Array* gradient) const {
        if (is_leaf_) { gradient_->add(gradient); } 
        else { expression_view_->backward(gradient); }
    }

    void print_gradient() {
        if (requires_gradient_) {
            for(auto i = 0; i < gradient_->size(); ++i) std::cout << gradient_->data()[i] << " ";
        }
    }

    void copy(const Array* other) final {
        Array::copy(other);
        requires_gradient_ = false;
        is_leaf_ = false;
    }

    void move(Array* other) final {
        Array::move(other);
        requires_gradient_ = false;
        is_leaf_ = false;
    }

    void copy(const Tensor* other) {
        Array::copy(other);
        requires_gradient_ = other->requires_gradient_;
        if (requires_gradient_) {
            gradient_ = new Array(other);
        }
    }

    void move(Tensor* other) {
        Array::move(other);
        requires_gradient_ = other->requires_gradient_;
        if (requires_gradient_) {
            gradient_ = other->gradient_;
            other->gradient_ = nullptr;
        }
    }

    void copy(const Tensor& other) { copy(&other); }
    void move(Tensor&& other) { move(&other); }

    bool requires_gradient() const { return requires_gradient_; }
    bool is_leaf() const { return is_leaf_; }

    void requires_gradient(bool status) {        

        if(status == true && requires_gradient_ == false) {
            requires_gradient_ = true;
            gradient_ = new Array(shape());
        }

        if(status == false && requires_gradient_ == true) {
            requires_gradient_ = false;
            delete gradient_;
            gradient_ = nullptr;
        }
    }

    void is_leaf(bool status) { is_leaf_ = status; }

    private:
    bool requires_gradient_;
    bool is_leaf_;
    Array* gradient_ = nullptr;
    Expression* expression_view_ = nullptr;
};

} // namespace internal

#endif // INTERNAL_TENSOR_HPP