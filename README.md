# CaberNet C++ Deep Learning Library

## Join the Discord:

https://discord.gg/aDxCxYEm


## To build the project

Please see the [contributing](.github/CONTRIBUTING.md#building-the-library) guide for more information.

## Description

This is a prototype for a full C++ deep learning library inspired by PyTorch API. It has one notable difference: when you perform an operation, the program doesn't actually execute it immediately. Instead, it allocates a node into a graph, waiting for you to call the perform() method on the result (like tensorflow but this is a dynamic graph). This allows the programmer to perform operations without making new memory allocations.

There is an example [here](examples/model.cpp) , of the digit MNIST dataset for a simple neural network working. Since I'm not a facebook team don't expect pytorch's performance, but it works nice.

In the future, I plan to re write the backend using static polymorphism to avoid the virtual calls that disables the compilers optimizations.

Example:

```cpp
#include <iostream>
#include <CaberNet.h>

int main() {

    net::Tensor<float> x({2,3}, net::requires_gradient::False); x.fill({1,2,3,4,5,6});
    net::Tensor<float> w({4,3}, net::requires_gradient::True); w.fill({1,2,-3,4,5,6,7,8,-9,10,11,-12});
    net::Tensor<float> v({2,4}, net::requires_gradient::True); v.fill({1,2,-3,4,5,6,7,8});

    // Or use just a boolean.
    net::Tensor<float> b({1,4}, true); b.fill({1,2,3,4});
    net::Tensor<float> I({2,4}, false); I.fill(1);

    x = net::function::linear(x,w,b);
    x = net::function::relu(x);
    x = x + net::matmul(v, w) * x;
    // builds an internal computational graph.

    x.perform();
    x.backward(I); // transverses the graph.

    std::cout << x << std::endl;
    std::cout << w.gradient() << std::endl;
    std::cout << b.gradient() << std::endl;
    return 0;
}
```

It produced the same results as the equivalent PyTorch code. 
You can also build layers like this: 

```cpp

#include <CaberNet.h>

struct Autoencoder : public net::Model<Autoencoder> {

    // it uses the CRTP pattern, so you define the forward method
    // to use the () operator when performing operations. 

    Autoencoder() {
        layers.configure_optimizer(optimizer);
    }

    net::layer::Sequence layers {
        net::layer::Linear(784, 128, net::initializer::He), // default initializer
        net::layer::ReLU(),
        net::layer::Linear(128, 5),
        net::layer::LogSoftmax(1/*axis*/)
    };

    net::Tensor<float> forward(net::Tensor<float> x) {
        x = layers(x);
        return x;
    }

    net::optimizer::SGD optimizer {/*learning rate*/ 0.1};

    void step() { optimizer.step(); }
};

int main() {
    Autoencoder network;
    net::Tensor<float> input({5,784}, true); input.fill(1) // fills with ones
    net::Tensor<float> output = network(input);
    net::Tensor<int> labels({5}); labels.fill({1,2,3,4,5});

    net::criterion::NLLLoss loss_function(output, labels);

    output.perform();

    std::cout << loss_function.loss() << std::endl;

    loss_function.backward(); // backpropagate the gradients
    network.step() // triggers the optimizer. 
}

```

Eigen library is used for performing all operations. The code is also backend-agnostic, meaning you can write your custom CUDA implementations if needed.

## Acknowledgements

Thanks for all your work!:

* @prince-chrismc. All your work is extremely valuable. 
