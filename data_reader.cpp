#include <iostream>
#include <fstream>
#include <vector>
#include <stdint.h>

struct IDXHeader {
    uint32_t magic_number;
    uint32_t number_of_images;
    uint32_t number_of_rows;
    uint32_t number_of_columns;
};

using quantized_int_type = uint8_t;
bool readIDXFile(const std::string& file_path, std::vector<std::vector<uint8_t>>& image_data) {
    std::ifstream file(file_path, std::ios::binary);
    
    if (!file) {
        std::cerr << "Failed to open file: " << file_path << std::endl;
        return false;
    }


    IDXHeader header;
    file.read(reinterpret_cast<char*>(&header), sizeof(header));
    
        if (header.magic_number != 0x00000803) {
        std::cerr << "Invalid IDX file format" << std::endl;
        return false;
    }

    const std::size_t image_size = header.number_of_rows * header.number_of_columns;
    image_data.resize(header.number_of_images, std::vector<uint8_t>(image_size));

    for (size_t i = 0; i < header.number_of_images; ++i) {
        file.read(reinterpret_cast<char*>(image_data[i].data()), image_size);
    }

    file.close();
    return true;
}


int main() {
    
    return 0;
}