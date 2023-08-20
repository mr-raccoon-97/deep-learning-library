#include "internal_array.hpp"
#include "internal_operations.hpp"
#include "internal_tensor.hpp"

#include "config.h"

#if defined(USING_EIGEN_BACKEND)
#include <eigen3/Eigen/Dense>

namespace internal {

// Array operations

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

// Expression operations
// Addition

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


// Expression operations
// Multiplication

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


// Expression operations
// Matrix multiplication

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

#endif