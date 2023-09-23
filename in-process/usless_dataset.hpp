#ifndef DATASET_H_
#define DATASET_H_

#include <iostream>
#include <vector>
#include <random>
#include <algorithm>
#include <numeric>
#include <memory>
#include <list>

#include "./linear_algebra.h"
#include "./normalizers.h"

class Dataset
{
    public:
    struct Batch {
        Matrix features;
        std::vector<int> targets;
        Batch(std::size_t batch_size, std::size_t features_size);
        void shuffle();
    };

    using iterator = typename std::list<Batch>::iterator;
    using const_iterator = typename std::list<Batch>::const_iterator;

    iterator begin();
    iterator end();
    const_iterator begin() const;
    const_iterator end() const;

    Dataset(
        const std::vector<std::vector<float>>& data,
        const std::vector<int>& targets,
        std::size_t batch_size,
        bool normalize = true,
        std::unique_ptr<Normalizer> normalizer = std::make_unique<Standard>()
    );

    void shuffle();

private:
    std::list<Batch> dataset_;
    std::size_t batch_size_;
    std::size_t features_size_;
    std::unique_ptr<Normalizer> normalizer_;
};


Dataset::Batch::Batch(std::size_t batch_size, std::size_t features_size)
    :   features(batch_size, features_size),
        targets(batch_size)
{}

void Dataset::Batch::shuffle()
{
    std::vector<std::size_t> indices(targets.size());
    std::iota(indices.begin(), indices.end(), 0);
    std::shuffle(indices.begin(), indices.end(), std::mt19937{std::random_device{}()});
    Matrix shuffled_features(features.rows(), features.cols());
    std::vector<int> shuffled_targets(targets.size());
    for(std::size_t i = 0; i < indices.size(); i++)
    {
        shuffled_features.row(i) = features.row(indices[i]);
        shuffled_targets[i] = targets[indices[i]];
    }
    features = shuffled_features;
    targets = shuffled_targets;
}

Dataset::iterator Dataset::begin(){ return dataset_.begin(); }
Dataset::iterator Dataset::end(){ return dataset_.end(); }

Dataset::const_iterator Dataset::begin() const{ return dataset_.begin(); }
Dataset::const_iterator Dataset::end() const{ return dataset_.end(); }


Dataset::Dataset(
    const std::vector<std::vector<float>>& features,
    const std::vector<int>& targets,
    std::size_t batch_size,
    bool normalize,
    std::unique_ptr<Normalizer> normalizer
) 
:   batch_size_(batch_size)
,   features_size_(features[0].size())
{
    if(normalize){
        normalizer_ = std::move(normalizer);
    }

    for(std::size_t i = 0; i < features.size()/batch_size; i++)
    {
        Batch batch(batch_size, features[0].size());
        for(std::size_t j = 0; j < batch_size; j++)
        {
            batch.features.row(j) = Eigen::Map<const Eigen::RowVectorXf>(
                (normalizer_->fit_transform(features[i * batch_size + j])).data(),
                features[0].size()
            );

            batch.targets[j] = targets[i * batch_size + j];
        }
        dataset_.emplace_back(std::move(batch));
    }
}

void Dataset::shuffle()
{
    for(auto& batch : dataset_)
    {
        batch.shuffle();
    }
}

#endif