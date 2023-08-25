#ifndef INTERNAL_OPERATION_ADDITION_H
#define INTERNAL_OPERATION_ADDITION_H

#include "internal_operation.hpp"

namespace internal {

class Array;
class Tensor;

class Addition : public Operation {
    public:
    ~Addition() final = default;
    Addition(const Tensor* first, const Tensor* second);

    void backward(Array* gradient) const final;
    Tensor perform() const final;
};

} // namespace internal 

#endif // INTERNAL_OPERATION_ADDITION_H