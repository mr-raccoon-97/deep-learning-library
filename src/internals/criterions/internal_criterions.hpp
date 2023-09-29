#ifndef INTERNAL_CRITERIONS_HPP
#define INTERNAL_CRITERIONS_HPP

#include "../internal_tensor.hpp"
#include "../internal_array.hpp"

namespace internal {

class Criterion {
    public:
    using size_type = Tensor::size_type;
    using shape_type = Tensor::shape_type;
    using scalar_type = Tensor::scalar_type;

    Criterion(Tensor* output, Array<size_type>* targets) {
        output_ = output;
        targets_ = targets;
    }

    virtual ~Criterion() = default;
    virtual scalar_type loss() const = 0;   

    Tensor* output() const { return output_; }
    Array<size_type>* targets() const { return targets_; }

    size_type number_of_classes() const { return output()->size() / batch_size(); }
    size_type batch_size() const { return output()->shape().front(); }

    private:
    Tensor* output_;
    Array<size_type>* targets_;
};

class NLLLoss : public Criterion {
    public:
    ~NLLLoss() final = default;
    NLLLoss(Tensor* output, Array<size_type>* targets) : Criterion(output, targets) {}
    scalar_type loss() const final;
};

} // namespace internal

#endif // INTERNAL_CRITERIONS_HPP