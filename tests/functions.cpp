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
    net::Tensor x({2,2}, { 5, 5, 5, 5 } , true);
    net::Tensor W({3,2}, { 2, 2, 2, 2, 2, 2 } ,true);
    net::Tensor b({1,3}, { 1, 1, 1 }, true);
    net::Tensor bb({2, 3}, { 1, 0, 1, 0, 1, 0 } , false);
    net::Tensor tt({2, 3}, { 2, 2, 2, 1, 2 ,3 }, false);

    x = net::function::linear(x,W,b);
    
    for (auto element : x) std::cout << element << " ";
}