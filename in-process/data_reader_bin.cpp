#include <iostream>
#include <fstream>
#include <vector>
#include <cstdint>
#include <string>
#include <list>

class Dataset {
    public:
    Dataset(std::size_t reserve_size = 60000) {
        features_.reserve(reserve_size);
        targets_.reserve(reserve_size);
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

            for (int i = 0; i < num_imgs; ++i) {
                file.seekg(16 + i * rows * cols, std::ios::beg);

                std::vector<uint8_t> image(rows * cols);
                file.read(reinterpret_cast<char*>(&image[0]), rows * cols);
                features_.emplace_back(std::move(image));
            }
        }
            
        else {
            std::cout << "Error opening file." << std::endl;
        }
    }

    void read_targets(std::string filepath) {
        std::ifstream file(filepath, std::ios::binary);
        if (file.is_open()) {
            uint32_t magic, num_labels;
            file.read(reinterpret_cast<char*>(&magic), 4);
            file.read(reinterpret_cast<char*>(&num_labels), 4);
 
            magic = swap_endian(magic);
            num_labels = swap_endian(num_labels);

            for (int i = 0; i < num_labels; ++i) {
                file.seekg(8 + i, std::ios::beg);

                uint8_t label;
                file.read(reinterpret_cast<char*>(&label), 1);
                targets_.push_back(label);
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

    std::vector<std::vector<uint8_t>> features_;
    std::vector<uint8_t> targets_;

    std::size_t batch_size_;
};

class Loader {
    public:
    Loader(std::size_t batch_size);
    

    private:
    std::size_t batch_size_;
};
 
int main() {
    Dataset data;
    data.read_features("data/train-images.idx3-ubyte");
    data.read_targets("data/train-labels.idx1-ubyte");
    return 0;
}