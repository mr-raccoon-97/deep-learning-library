#ifndef INTERNAL_CRITERIONS_HPP
#define INTERNAL_CRITERIONS_HPP

#include "../internal_tensor.hpp"
#include "../internal_array.hpp"

namespace internal {

// TODO : manage the int type.

class Criterion {
    public:
    using size_type = Tensor::size_type;
    using shape_type = Tensor::shape_type;
    using scalar_type = Tensor::scalar_type;

    Criterion(Tensor* output, Array<int>* targets) {
        output_ = output;
        targets_ = targets;
        gradient_ = std::make_unique<Tensor>(output_->shape(), false);
    }

    virtual ~Criterion() = default;
    virtual scalar_type loss() const = 0;   
    virtual void backward() = 0;

    Tensor* output() const { return output_; }
    Array<int>* targets() const { return targets_; }

    size_type number_of_classes() const { return output()->size() / batch_size(); }
    size_type batch_size() const { return output()->shape().front(); }

    Tensor* gradient() const { return gradient_.get(); }

    private:
    Tensor* output_;
    std::unique_ptr<Tensor> gradient_;
    Array<int>* targets_;
};

class NLLLoss : public Criterion {
    public:
    ~NLLLoss() final = default;
    NLLLoss(Tensor* output, Array<int>* targets) : Criterion(output, targets) {}
    scalar_type loss() const final;
    void backward() final;
};

} // namespace internal

#endif // INTERNAL_CRITERIONS_HPP