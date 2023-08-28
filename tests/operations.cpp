#include <Cabernet/tensor.h>

int main() {
    net::Tensor x({2,2}, true, true); for (auto& e : x) e = 1;
    net::Tensor y({2,2}, true, true); for (auto& e : y) e = -3;
    net::Tensor z({2,2}, true, true); for (auto& e : z) e = 4;
    net::Tensor I({2,2}); for (auto& e : I) e = 1;

    net::Tensor result = x * z + y * z;
    result.backward(I);

    x.print_gradient(); 
    std::cout << std::endl; 
    y.print_gradient();

    std::cout << std::endl; 
    z.print_gradient();
}

