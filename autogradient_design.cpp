#include <iostream>
#include <Cabernet/tensor.h>

int main() {
    net::Tensor x({2, 2}, true, true);
    for (auto& e : x) e = 1;
    for (auto e : x) std::cout << e << std::endl;
}