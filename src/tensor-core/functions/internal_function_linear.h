#ifndef INTERNAL_FUNCTION_LINEAR_H
#define INTERNAL_FUNCTION_LINEAR_H

#include "../internal_expression.hpp"
#include "../internal_types.h"

namespace internal {

class Linear : public Expression {
    public:
    ~Linear() final;
    Linear(const Tensor* input, const Tensor* weight, const Tensor* bias);
    Tensor perform() const final;
    void backward(Array* gradient) const final;

    const Tensor* input() const;
    const Tensor* weight() const;
    const Tensor* bias() const;   

    type::size_type rows_dimension() const;
    type::size_type columns_dimension() const;
    type::size_type inner_dimension() const;

    bool gradient_requirement() const;

    private:
    const Tensor* input_;
    const Tensor* weight_;
    const Tensor* bias_;       
};

}

#endif // INTERNAL_FUNCTION_LINEAR_H