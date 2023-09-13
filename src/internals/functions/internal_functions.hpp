#ifndef INTERNAL_FUNCTIONS_HPP
#define INTERNAL_FUNCTIONS_HPP

#include "../internal_tensor.hpp"
#include "../internal_expression.hpp"

namespace internal {

class Function : public Expression {
    public:
    ~Function() override = default;
    Function(Tensor* input) { input_ = input; }
    Tensor* input() const { return input_; }

    private:
    Tensor* input_;
};

class Linear : public Function {
    public:
    ~Linear() final = default;
    Linear(Tensor* input, Tensor* weight, Tensor* bias);
    Tensor* forward() final;
    void backward(Array* gradient) const final;

    Tensor* weight() const { return weight_; }
    Tensor* bias() const { return bias_; }   

    size_type rows_dimension() const { return input()->shape().front(); }
    size_type inner_dimension() const { return input()->shape().back(); }
    size_type columns_dimension() const { return weight()->shape().front(); }

    private:
    Tensor* weight_;
    Tensor* bias_;       
};

class ReLU : public Function {
    public:
    ~ReLU() final = default;
    ReLU(Tensor* input);
    Tensor* forward() final;
    void backward(Array* gradient) const final;
};

class Softmax : public Function {
    public:
    static void inplace(Tensor* input, int axis);
};

class LogSoftmax : public Function {
    public:
    static void inplace(Tensor* input, int axis);
};

} // namespace internal

#endif // INTERNAL_FUNCTIONS_HPP