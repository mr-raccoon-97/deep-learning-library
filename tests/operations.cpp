#include <CaberNet/CaberNet.h>

int main() {
    net::Tensor x({2,3}, false); x.fill({1,2,3,4,5,6});
    net::Tensor y({2,3}, true); y.fill({1,1,1,-1,-1,-1});
    net::Tensor z({2,3}, true); z.fill(1);  
    net::Tensor I({2,3}, false); I.fill(1);
    net::Tensor w({3,3}, true); w.fill({1,2,3,4,5,6,7,8,9});

    x = net::matmul(x, w);
    x =  x * z + y * z + z * y;
    x.perform();
    x.backward(I);

    std::cout << "x : " << x << std::endl; 
    std::cout << "Jy: " << y.gradient() << std::endl;
    std::cout << "Jz: " << z.gradient() << std::endl;
    std::cout << "Jw: " << w.gradient() << std::endl;
}