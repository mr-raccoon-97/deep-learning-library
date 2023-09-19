#include <Cabernet/tensor.h>
#include <Cabernet/functions.h>

#include <iostream>
#include <vector>

/*

g++ functions.cpp -LCabernet/lib -lCabernet -I Cabernet/include

import torch
import torch.nn.functional as F

# Define tensors
x = torch.tensor([[-1, 2], [5, 1]], dtype=torch.float32, requires_grad=False)
W = torch.tensor([[2, -2], [2, 2]], dtype=torch.float32, requires_grad=True)
b = torch.tensor([[-10, -2]], dtype=torch.float32, requires_grad=True)
I = torch.tensor([[1, 1], [1, 1]], dtype=torch.float32)

# Perform linear operation using torch.nn.functional.linear
x = F.linear(x, W, b)
x = F.relu(x)
x = F.linear(x, W, b)

# Backpropagation
x.backward(I)


# Print gradients
print(x)
print("Gradient of W:")
print(W.grad)
print("Gradient of b:")
print(b.grad)

*/

int main() {
    net::Tensor x({2,2}, { -1, 2, 5, 1 } , false);
    net::Tensor W({2,2}, { 2, -2, 2, 2 } ,true);
    net::Tensor b({1,2}, { -10, -2 }, true);
    net::Tensor I({2,2}, { 1, 1, 1, 1 }, false);

    x = net::function::linear(x,W,b);
    x = net::function::relu(x);
    x = net::function::linear(x,W,b);

    x.perform();

    for (auto element : x) std::cout << element << std::endl;

    x.backward(I);

    std::cout << "Gradient of W:" << std::endl;
    for (auto element : W.gradient()) std::cout << element << std::endl;

    std::cout << "Gradient of b:" << std::endl;
    for (auto element : b.gradient()) std::cout << element << std::endl;

    return 0;
}