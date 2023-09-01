#include <iostream>
#include <vector>
#include <span>

// experimental c++20
// to replace the perform method return std::unique_ptr type for a View that has optional ownership.

struct Tensor {
    Tensor(){
        data.resize(5);
        for (auto& e : data) e = 1;
    }
    std::vector<float> data;
    void print() {
        for ( auto e : data) std::cout << e;
    }
};

struct View {
    View(const Tensor& tensor)
    : data(tensor.data) {}

    std::span<const float> data;

    void print() {
        for ( auto e : data) std::cout << e;
    }
};

int main() {
    Tensor t;
    View v(t);
    v.print();
}