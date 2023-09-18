#include <iostream>
#include <memory>

int main() {
    /*
    Net::Tensor x({2, 2}, true);  // is leaf here, but don't requires gradient
    Net::Tensor W({2, 2}, true);
    Net::Tensor b({2, 2}, true);
    
    x = Net::function::linear(x, W, b); 
    x = Net::function::relu(x);
    x = Net::function::linear(x, W, b);
    x = cnet::function::softmax(x);

    x.perform();
    x.derive();

    std::cout << x.gradient();  

    */
}