/*************************************************************************************************\

This is the main data structure of the library. It acts as a node of the computational graph. It is
provided with virtual forward and backward methods that are used to perform the forward and backward 
passes of the data through the graph. Those methods are mean to be overriden when implementing 
different operations or functions as nodes of the computational graph.

It is a multidimensional array that stores the metadata of the tensors and the gradients of
the tensors. 

The requires_gradient_ flag is used to determine if the tensor needs to store the gradient or not.

The is_leaf_ flag is used to determine if the tensor is a leaf node of the computational graph or
not. The Tensor class is leaf by default.

The gradient of the tensor is stored in the gradient_ pointer. If the tensor is a leaf
node, then the Tensor class owns the gradient_ pointer and is responsible for its deletion, otherwise
the gradient_ pointer is just a reference to the gradient of the real owner of the gradient, and should
not be deleted by the Tensor class. Since optional ownership cannot be expressed with smart pointers, the
gradient_ pointer is a raw pointer.

The gradient of the tensor is created only when the requires_gradient_ flag is set to true by the 
requires_gradient() method. The gradient_ pointer is set to nullptr by default.

/*************************************************************************************************/

#ifndef INTERNAL_TENSOR_HPP
#define INTERNAL_TENSOR_HPP

#include <iostream>
#include <vector>
#include <memory>

#include "internal_array.hpp"

namespace internal {

class Tensor : public Array {
    public:
    Tensor(bool leaf_status = true)
    :   Array() {
        is_leaf_ = leaf_status; 
        requires_gradient_ = false;
    }

    Tensor(shape_type shape, bool gradient_requirement = false, bool leaf_status = true) 
    :   Array(shape) {
        is_leaf_ = leaf_status; 
        requires_gradient_ = gradient_requirement;
        if (is_leaf_ && requires_gradient_) gradient_ = new Array(shape);
    }

    Tensor(const Tensor* other) { copy(other); }
    Tensor(Array&& other) { Array::move(&other); }

    ~Tensor() override { if (is_leaf_ && requires_gradient_) delete gradient_; }
    Tensor(const Tensor& other) = delete;
    Tensor(Tensor&& other) = delete;
    Tensor& operator=(const Tensor& other) = delete;
    Tensor& operator=(Tensor&& other) = delete;


    void copy(const Tensor* other) {
        Array::copy(other);
        requires_gradient_ = other->requires_gradient_;

        if (requires_gradient_ ) {
            if (other->is_leaf_ && is_leaf_) {
                if (!gradient_) gradient_ = new Array(other->gradient_);
                else gradient_->copy(other->gradient_);
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
    bool requires_gradient() const { return requires_gradient_; }

    virtual Tensor* forward() { return this; }
    virtual void backward(Array* gradient) const { gradient_->add(gradient); }

    protected:
    bool is_leaf_;
    bool requires_gradient_;

    private:
    Array* gradient_;
};

} // namespace internal

#endif // INTERNAL_TENSOR_HPP