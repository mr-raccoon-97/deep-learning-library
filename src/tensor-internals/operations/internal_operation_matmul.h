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

    std::unique_ptr<Tensor> perform() const final;
    void backward(Array* gradient) const final;

    size_type rows_dimension() const;
    size_type columns_dimension() const;
    size_type inner_dimension() const;

    protected:
    size_type rows_dimension_;
    size_type columns_dimension_;
    size_type inner_dimension_;
};

} // namespace internal 

#endif // INTERNAL_OPERATION_MATMUL_H