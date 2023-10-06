#ifndef INTERNAL_BASE_HPP
#define INTERNAL_BASE_HPP

#include <iostream>
#include <vector>

namespace internal {

class Base {
    public:
    using size_type = std::size_t;
    using shape_type = std::vector<size_type>;

    Base() = default;

    Base(shape_type shape) : shape_(shape) {
        size_ = 1;
        for (auto& dimension : shape_) size_ *= dimension; 
    }

    size_type size() const { return size_; }
    shape_type shape() const { return shape_; }
    size_type rank() const { return shape_.size(); }

    void reshape(shape_type shape) {
        shape_ = shape;
        size_ = 1;
        for (auto& dimension : shape_) size_ *= dimension; 
    }

    void melt() {
        shape_.clear();
        shape_.push_back(size_);
    }

    void collapse() {
        size_ = 0;
        shape_.clear();
    }

    private:
    size_type size_;
    shape_type shape_;
    // shape_type strides_;
};

}

#endif