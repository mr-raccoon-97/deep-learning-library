/*We will need to automate this with CMake. I didn't decide yet what unit test library to use
If someone wants to do it will be very appreciated.

g++ operations.cpp -LCaberNet/lib -lCaberNet -I CaberNet/include
./a.out
*/

#include <iostream>
#include <CaberNet/CaberNet.h>

int main() {
    net::Tensor x({2,3}, false); x.fill({1,2,3,4,5,6});
    net::Tensor y({2,3}, true);  y.fill({1,1,1,-1,-1,-1});
    net::Tensor z({2,3}, true);  z.fill(1);  
    net::Tensor I({2,3}, false); I.fill(1);
    net::Tensor w({3,3}, true);  w.fill({1,2,3,4,5,6,7,8,9});

    x = x + I;
    x = net::matmul(x, w);
    x =  x * z + y * z + z * y;
    x.perform();
    x.backward(I);

    std::cout << "x : " << x << std::endl; 
    std::cout << "Jy: " << y.gradient() << std::endl;
    std::cout << "Jz: " << z.gradient() << std::endl;
    std::cout << "Jw: " << w.gradient() << std::endl;
}

/*

Results should be:
x : [44, 53, 62, 76, 94, 112]
Jy: [2, 2, 2, 2, 2, 2]
Jz: [44, 53, 62, 76, 94, 112]
Jw: [7, 7, 7, 9, 9, 9, 11, 11, 11]

*/

/*
import torch

# Initialize tensors
x = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float32, requires_grad=False)
y = torch.tensor([[1, 1, 1], [-1, -1, -1]], dtype=torch.float32, requires_grad=True)
z = torch.tensor([[1, 1, 1], [1, 1, 1]], dtype=torch.float32, requires_grad=True)
I = torch.tensor([[1, 1, 1], [1, 1, 1]], dtype=torch.float32, requires_grad=False)
w = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=torch.float32, requires_grad=True)


# Perform operations

x = x + I
x = torch.matmul(x, w)
x = x * z + y * z + z * y
x.backward(I)

# Print results
print("x : ", x)
print("Jy: ", y.grad)
print("Jz: ", z.grad)
print("Jw: ", w.grad)
*/