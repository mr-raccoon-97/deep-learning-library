#include <CaberNet/CaberNet.h>


int main() {
    net::Tensor<float> X({2,2}, true); X.fill(1);
    net::Tensor<float> I({2,2}); I.fill(1);
    X.backward(I);
    std::cout << X.gradient();

    net::optimizer::SGD optimizer(0.1);
    optimizer.add_parameter(X.internal());
    optimizer.step();
    std::cout << X;
    std::cout << X.gradient();
}