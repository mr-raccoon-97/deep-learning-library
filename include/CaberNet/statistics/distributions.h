#pragma once

#include <iostream>
#include <vector>
#include <random>
#include <memory>

namespace net::distribution {

template<typename T>
class Distribution {
    public:
    using real_type = T;
    virtual ~Distribution() = default;
    virtual real_type generate() = 0;

    protected:
    Distribution() = default;
};

template<typename T>
struct Normal : public Distribution<T> {
    public:
    using real_type = T;
    Normal(real_type mean, real_type standard_deviation)
    :   distribution_(mean, standard_deviation)
    ,   generator_(std::random_device()()) {}

    real_type generate() final { return distribution_(generator_); }

    std::normal_distribution<real_type> distribution_;
    std::mt19937 generator_;
};

} // namespace distribution