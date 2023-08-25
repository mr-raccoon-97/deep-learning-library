#ifndef INTERNAL_OPERATION_MATMUL_H
#define INTERNAL_OPERATION_MATMUL_H

#include "internal_operation.hpp"

namespace internal {

class Tensor;
class Array;

class Matmul : public Operation {
    public:
    ~Matmul() final = default;
    Matmul(const Tensor* first, const Tensor* second);

    Tensor perform() const final;
    void backward(Array* gradient) const final;

    protected:
    size_type rows_dimension;
    size_type columns_dimension;
    size_type inner_dimension;
};

} // namespace internal 

#endif // INTERNAL_OPERATION_MATMUL_H