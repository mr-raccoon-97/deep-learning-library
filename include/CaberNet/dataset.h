#include <iostream>
#include <fstream>
#include <vector>
#include <cstdint>
#include <string>
#include <list>
#include <algorithm>

#include "tensor.h"

namespace net {

class Dataset {
    public:
    
    Dataset(std::size_t batch_size, bool shuffle = false, std::size_t reserve_size = 60000) {
        batch_size_ = batch_size;
        shuffle_ = shuffle;
    }

    void read_targets(std::string filepath) {
        std::ifstream file(filepath, std::ios::binary);
        if (file.is_open()) {
            uint32_t magic, num_labels;
            file.read(reinterpret_cast<char*>(&magic), 4);
            file.read(reinterpret_cast<char*>(&num_labels), 4);
 
            magic = swap_endian(magic);
            num_labels = swap_endian(num_labels);

            for(std::size_t batch = 0; batch < num_labels; batch += batch_size_) {
                targets_.emplace_back(Tensor<int>({batch_size_}));
                for(std::size_t index = 0; index < batch_size_; ++index) {
                    uint8_t label;
                    file.read(reinterpret_cast<char*>(&label), 1);
                    targets_.back().data()[index] = static_cast<int>(label);
                };
                std::cout << targets_.back() << std::endl;
            }
        }
        
        else {
            std::cout << "Error opening file." << std::endl;
        }

    }


    void read_features(std::string filepath) {
        std::ifstream file(filepath, std::ios::binary);
        if (file.is_open()) {
            uint32_t magic, num_imgs, rows, cols;
            file.read(reinterpret_cast<char*>(&magic), 4);
            file.read(reinterpret_cast<char*>(&num_imgs), 4);
            file.read(reinterpret_cast<char*>(&rows), 4);
            file.read(reinterpret_cast<char*>(&cols), 4);
 
            magic = swap_endian(magic);
            num_imgs = swap_endian(num_imgs);
            rows = swap_endian(rows);
            cols = swap_endian(cols);

            features_size_ = rows * cols;

            for (std::size_t batch = 0; batch < num_imgs; batch += batch_size_) {
                features_.emplace_back(Tensor<float>({batch_size_, features_size_}, false, true));
                for (std::size_t index = 0; index < batch_size_; ++index) {
                    std::vector<uint8_t> features(features_size_);
                    file.read(reinterpret_cast<char*>(&features[0]), features_size_);
                    std::transform(
                        features.begin(),
                        features.end(), features_.back().data() + index * features_size_,
                        [](uint8_t x) { return static_cast<float>(x) / 255.0f; }
                    );
                }
                std::cout << features_.back() << std::endl;
            }

        }
            
        else {
            std::cout << "Error opening file." << std::endl;
        }
    }

    void clear() {
        features_.clear();
        targets_.clear();
    }

    private:
    uint32_t swap_endian(uint32_t val) {
        return ((val << 24) & 0xff000000) |
               ((val <<  8) & 0x00ff0000) |
               ((val >>  8) & 0x0000ff00) |
               ((val >> 24) & 0x000000ff);
    }

    std::vector<Tensor<float>> features_;
    std::vector<Tensor<int>> targets_;

    std::size_t features_size_;
    std::size_t batch_size_;
    bool shuffle_;
};
 
} // namespace net
