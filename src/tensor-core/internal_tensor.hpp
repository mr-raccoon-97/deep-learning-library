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

} // namespace internal

#endif // INTERNAL_TENSOR_HPP