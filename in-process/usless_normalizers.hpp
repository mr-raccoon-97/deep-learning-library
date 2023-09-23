#ifndef NORMALIZERS_H_
#define NORMALIZERS_H_

#include <vector>
#include <algorithm>
#include <numeric>
#include <cmath>

class Normalizer
{
public:
    virtual ~Normalizer() = default;

    virtual void fit(const std::vector<float>& features_vector);
    virtual std::vector<float> transform(const std::vector<float>& features_vector);
    virtual std::vector<float> fit_transform(const std::vector<float>& features_vector);
    virtual std::vector<float> inverse_transform(const std::vector<float>& features_vector);
};

class Standard : public Normalizer
{
public:
    Standard() = default;
    ~Standard() = default;

    void fit(const std::vector<float>& features_vector) override;
    std::vector<float> transform(const std::vector<float>& features_vector) override;
    std::vector<float> fit_transform(const std::vector<float>& features_vector) override;
    std::vector<float> inverse_transform(const std::vector<float>& features_vector) override;

private:
    float mean_;
    float standard_deviation_;
};


void Standard::fit(const std::vector<float>& features_vector) {
    mean_ = mean(features_vector);
    standard_deviation_ = standard_deviation(features_vector);
}

std::vector<float> Standard::transform(const std::vector<float>& features_vector)
{
    std::vector<float> transformed_features(features_vector.size());
    std::transform(
        features_vector.begin(),
        features_vector.end(),
        transformed_features.begin(),
        [this](float feature){ return (feature - mean_) / standard_deviation_; }
    );
    return transformed_features;
}

std::vector<float> Standard::fit_transform(const std::vector<float>& features_vector) {
    fit(features_vector);
    return transform(features_vector);
}

std::vector<float> Standard::inverse_transform(const std::vector<float>& features_vector)
{
    std::vector<float> inverse_transformed_features(features_vector.size());
    std::transform(
        features_vector.begin(),
        features_vector.end(),
        inverse_transformed_features.begin(),
        [this](float feature){return (feature * standard_deviation_) + mean_;}
    );
    return inverse_transformed_features;
}

#endif



