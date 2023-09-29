/*

To run this code build the library:
in your terminal:

https://github.com/xEricCardozo/CaberNet-cpp.git

and inside the folder:

mkdir build
cd build
cmake ..
make

then compile this file with:

g++ functions.cpp -LCaberNet/lib -lCaberNet -I CaberNet/include
./a.out

*/

#include <iostream>
#include <CaberNet/CaberNet.h>

int main() {
    // You can use enums to set the gradient requirement:
    net::Tensor x({2,3}, net::requires_gradient::False); x.fill({1,2,3,4,5,6});
    net::Tensor w({4,3}, net::requires_gradient::True); w.fill({1,2,-3,4,5,6,7,8,-9,10,11,-12});
    
    // Or use just a boolean. Whatever you prefer.
    net::Tensor b({1,4}, true); b.fill({1,2,3,4});
    net::Tensor I({2,4}, false); I.fill(1);

    x = net::function::linear(x,w,b);
    x = net::function::relu(x);
    x.perform();
    x.backward(I);

    std::cout << x << std::endl;
    std::cout << w.gradient() << std::endl;
    std::cout << b.gradient() << std::endl;
    return 0;
}

/*
Results should be:
[0, 34, 0, 0, 0, 79, 17, 27, ]
[0, 0, 0, 5, 7, 9, 4, 5, 6, 4, 5, 6, ]
[0, 2, 1, 1, ]

*/

/*

Equivalent pytorch code:

import torch.nn.functional as F

# Initialize tensors
x = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float32, requires_grad=False)
w = torch.tensor([[1, 2, -3], [4, 5, 6], [7, 8, -9], [10, 11, -12]], dtype=torch.float32, requires_grad=True)
b = torch.tensor([1, 2, 3, 4], dtype=torch.float32, requires_grad=True)
I = torch.tensor([[1,1,1,1],[1,1,1,1]], dtype=torch.float32)

# Perform linear operation using torch.nn.functional.linear
x = F.linear(x, w, b)
x = F.relu(x)
x.backward(I)

# Print the result
print(x)
print(w.grad)
print(b.grad)

*/