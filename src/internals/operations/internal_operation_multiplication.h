#ifndef INTERNAL_OPERATION_MULTIPLICATION_H
#define INTERNAL_OPERATION_MULTIPLICATION_H

#include "internal_operation.hpp"

namespace internal {

class Array;
class Tensor;

class Multiplication : public Operation {
    public:
    ~Multiplication() final = default;
    Multiplication(const Tensor* first, const Tensor* second);

    void backward(Array* gradient) const final;
    std::unique_ptr<Tensor> perform() const final; 
};

} // namespace internal 

#endif // INTERNAL_OPERATION_MULTIPLICATION_H