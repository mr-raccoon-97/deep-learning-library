#include <iostream>
#include <fstream>
#include <vector>
#include <cstdint>
 
uint32_t swap_endian(uint32_t val) {
    return ((val << 24) & 0xff000000) |
           ((val <<  8) & 0x00ff0000) |
           ((val >>  8) & 0x0000ff00) |
           ((val >> 24) & 0x000000ff);
}
 
void read_mnist_images(const std::string& filepath) {
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
 
        // Navigate to the image at index 5
        file.seekg(16 + 5 * rows * cols, std::ios::beg);
 
        // Read and print image at IDX=5
        std::vector<uint8_t> image(rows * cols);
        file.read(reinterpret_cast<char*>(&image[0]), rows * cols);
 
        std::cout << "Image at IDX=5:" << std::endl;
        for(int r = 0; r < rows; ++r) {
            for(int c = 0; c < cols; ++c) {
                std::cout << static_cast<int>(image[r * cols + c]) << ' ';
            }
            std::cout << std::endl;
        }
 
    } else {
        std::cout << "Error opening file." << std::endl;
    }
}
 
void read_mnist_labels(const std::string& filepath) {
    std::ifstream file(filepath, std::ios::binary);
    if (file.is_open()) {
        uint32_t magic, num_labels;
        file.read(reinterpret_cast<char*>(&magic), 4);
        file.read(reinterpret_cast<char*>(&num_labels), 4);
 
        magic = swap_endian(magic);
        num_labels = swap_endian(num_labels);
 
        // Navigate to the label at index 5
        file.seekg(8 + 5, std::ios::beg);
 
        // Read and print label at IDX=5
        uint8_t label;
        file.read(reinterpret_cast<char*>(&label), 1);
 
        std::cout << "Label at IDX=5: " << static_cast<int>(label) << std::endl;
 
    } else {
        std::cout << "Error opening file." << std::endl;
    }
}
 
int main() {
    read_mnist_images("train-images.idx3-ubyte");
    read_mnist_labels("train-labels.idx1-ubyte");
    return 0;
}