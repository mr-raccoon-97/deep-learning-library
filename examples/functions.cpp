/*
To test this code run:
g++ functions.cpp -LCaberNet/lib -lCaberNet -I CaberNet/include

And run this into a colab notebook in your browser (https://colab.research.google.com/):

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

#include <iostream>
#include <vector>
#include <CaberNet/CaberNet.h>


int main() {

    /* OUTDATED, new fill method should be added for initializer lists*/
    net::Tensor x({2,2}, { -1, 2, 5, 1 } , false);
    net::Tensor W({2,2}, { 2, -2, 2, 2 } ,true);
    net::Tensor b({1,2}, { -10, -2 }, true);
    net::Tensor I({2,2}, { 1, 1, 1, 1 }, false);

    x = net::function::linear(x,W,b);
    x = net::function::relu(x);
    x = net::function::linear(x,W,b);
    x = W * x + W;

    x.perform();
    x.backward(I);

    std::cout << "x:" << std::endl;
    for (auto element : x) std::cout << element << std::endl;

    std::cout << "Gradient of W:" << std::endl;
    for (auto element : W.gradient()) std::cout << element << std::endl;

    std::cout << "Gradient of b:" << std::endl;
    for (auto element : b.gradient()) std::cout << element << std::endl;

    return 0;
}
