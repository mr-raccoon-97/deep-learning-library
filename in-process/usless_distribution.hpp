#ifndef DISTRIBUTION_HPP
#define DISTRIBUTION_HPP

#include <iostream>
#include <vector>
#include <random>
#include <memory>

namespace net::distribution {

class Distribution {
    public:
    using real_type = float;
    virtual ~Distribution() = default;
    virtual std::vector<real_type> generate(std::size_t) = 0;

    protected:
    Distribution() = default;
};

class Normal : public Distribution {
    public:

    ~Normal() override = default;

    Normal(real_type mean, real_type standard_deviation)
    :   distribution_(mean, standard_deviation)
    {}

    std::vector<real_type> generate(std::size_t number_of_samples) override {
        std::vector<real_type> samples(number_of_samples);
        for(auto& sample : samples) sample = distribution_(_generator);
        return samples;
    }

    private:
    std::normal_distribution<real_type> distribution_;
    std::mt19937 _generator{std::random_device()()};
};

class Zeros : public Distribution {
    public:

    ~Zeros() override = default;
    Zeros() = default;

    std::vector<real_type> generate(std::size_t number_of_samples) override {
        std::vector<real_type> samples(number_of_samples);
        for(auto& sample : samples) sample = 0;
        return samples;
    }
};

} // namespace net::distribution

#endif // DISTRIBUTION_HPP