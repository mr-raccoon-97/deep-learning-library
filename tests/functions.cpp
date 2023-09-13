#include <Cabernet/tensor.h>
#include <Cabernet/functions.h>

#include <iostream>
#include <vector>

/*
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
    net::Tensor x({2,2}, true); for (auto& e : x) e = 5;
    net::Tensor W({3,2}, true); for (auto& e : W) e = 2;
    net::Tensor b({1,3}, true); for (auto& e : b) e = 1;
    net::Tensor bb({2, 3}, false);
    net::Tensor tt({2, 3}, false);

    std::vector<float> bb_vec = {1, 0, 1, 0, 1, 0};
    std::vector<float> tt_vec = {2, 2, 2, 1, 2, 3};

    int i = 0;
    for (auto& e : bb) {
        e = bb_vec[i];
        i++;
    }

    i = 0;

    for (auto& e : tt) {
        e = tt_vec[i];
        i++;
    }

    net::Tensor p = net::function::linear(x, W, b);
    net::function::log_softmax(p, 0);
    net::Tensor result = (p + bb) * tt;

    result.perform();

    for (auto& e : result) {
        std::cout << e << " ";
    } 
}