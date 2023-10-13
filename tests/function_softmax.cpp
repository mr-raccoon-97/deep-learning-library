#include <CaberNet/CaberNet.h>

/*

x = torch.tensor([[1,2,3],[-1,3,2]], dtype = torch.float32, requires_grad = True)
y = F.log_softmax(x, dim = 0)
I = torch.tensor([[1,3,1],[1,2,1]], dtype = torch.float32)
print(y)

y.backward(I)

print(x.grad)


*/

int main() {
    /*
    net::Tensor<float> x({2, 3}, true); x.fill({1,2,3,-1,3,2});
    net::Tensor<float> I({2, 3}); I.fill({1,3,1,1,2,1});

    net::Tensor<float> y({2,3});

    y = net::function::log_softmax(x, 1);
    y.perform();
    y.backward(I);

    std::cout << "x: " << x << std::endl;
    std::cout << "y: " << y << std::endl;
    std::cout << "x.grad: " << x.gradient() << std::endl;
    */

    net::Tensor<float> x({2,3}, false); x.fill({1,2,3,4,5,6});
    net::Tensor<float> w({4,3}, true); w.fill({1,2,-3,4,5,6,7,8,-9,10,11,-12});
    net::Tensor<float> b({1,4}, true); b.fill({1,2,3,4});
    net::Tensor<float> I({2,4}, false); I.fill(1);

    x = net::function::linear(x,w,b);
    x = net::function::relu(x);
    x = net::function::log_softmax(x, 1);

    x.perform();
    x.backward(I);

    std::cout << "x: " << x << std::endl;
    std::cout << "w.grad: " << w.gradient() << std::endl;
    std::cout << "b.grad: " << b.gradient() << std::endl;
}