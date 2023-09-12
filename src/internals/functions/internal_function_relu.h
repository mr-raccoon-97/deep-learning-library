#ifndef INTERNAL_FUNCTION_RELU_H
#define INTERNAL_FUNCTION_RELU_H

#include "../internal_expression.hpp"

namespace internal {

class ReLU : public Expression {
    public:
    ~ReLU() final = default;
    ReLU(Tensor* input);

    const Tensor* result() const;
    
    void backward(Array* gradient) const final;    

    private:
    Tensor* result_;
};

}

#endif // INTERNAL_FUNCTION_RELU_H