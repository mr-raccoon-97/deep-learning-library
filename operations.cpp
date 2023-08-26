#include <Cabernet/tensor.h>


int main() {
    net::Tensor x({2,2}, true, true); for (auto& e : x) e = 2;
    net::Tensor y({2,2}, true, true); for (auto& e : y) e = 3;
    net::Tensor I({2,2})

    net::Tensor z = x + y;
    for (auto e : z) std::cout << e;

}