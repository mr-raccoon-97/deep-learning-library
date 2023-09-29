#include <CaberNet/CaberNet.h>
#include <iostream>

int main() {
    net::Subscripts y({2, 3, 4}); y.fill(1);
    std::cout << y << std::endl; 

    return 0;
}