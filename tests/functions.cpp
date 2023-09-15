#include <Cabernet/tensor.h>
#include <Cabernet/functions.h>

#include <iostream>
#include <vector>

/*

g++ 

import torch
import torch.nn.functional as F

x = torch.tensor([
    [5.,5.],
    [5.,5.]
], requires_grad = True)

W = torch.tensor([
    [2.,2.],
    [2.,2.],
    [2.,2.]
], requires_grad = True)

b = torch.tensor([
    [1.,1.,1.]
], requires_grad = True)

bb = torch.tensor([
    [1, 0 ,1],
    [0, 1, 0]
])

tt = torch.tensor([
    [2, 2, 2],
    [1, 2, 3]
])


a = F.linear(x,W,b)
b = F.log_softmax(a, dim = 0)
c = (b + bb) * tt
result = F.relu(c)

print(result)
*/

int main() {
    net::Tensor x({2,2}, { -1, 2, 5, 1 } , false);
    net::Tensor W({2,2}, { 2, -2, 2, 2 } ,true);
    net::Tensor b({1,2}, { -10, -2 }, true);

    x = net::function::linear(x,W,b);
    x = x + W;
    x = net::function::linear(x,W,b);
    x = net::function::relu(x);
    
    x.perform();

    for (auto element : x) std::cout << element << " ";

    return 0;
}