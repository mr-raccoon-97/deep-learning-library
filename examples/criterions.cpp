#include <CaberNet.h>


int main() {
    net::layer::Linear linear(5, 3, net::initializer::He);

    net::Tensor<float> input({1, 5}); input.fill(net::initializer::He);
    net::Tensor<float> output = linear(input);
    net::Tensor<int> targets({1}); targets.fill(0);

    net::criterion::NegativeLogLikelihood loss(output, targets);

    net::optimizer::SGD optimizer(0.1);
    
    linear.set_optimizer(optimizer.get());
    

    std::cout << loss.loss() << std::endl;
    loss.backward();

    optimizer.step();
}