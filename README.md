# CaberNet C++ Deep Learning Library

## Join the Discord:

https://discord.gg/aDxCxYEm

## Description

If you want to contribute but don't understand my code, don't be afraid to ask, write in the discord server or send me an email : eric.m.cardozo@gmail.com

The code isn't fully documented yet, but it's very readable and structured in a highly decoupled manner, so it can grow indefinitely. The core part, which includes things related to the internal representation of tensors, the computational graph, and design, is now functional.

The API is currently inspired by PyTorch, with one notable difference: when you perform an operation, the program doesn't actually execute it immediately. Instead, it allocates a node into a graph, waiting for you to call the perform() method on the result ( like tensorflow but this is a dynamic graph ).Here's an example I created to test it:

```cpp
#include <iostream>
#include <CaberNet/CaberNet.h>

int main() {

    // You can use enums to set the gradient requirement:
    net::Tensor<float> x({2,3}, net::requires_gradient::False); x.fill({1,2,3,4,5,6});
    net::Tensor<float> w({4,3}, net::requires_gradient::True); w.fill({1,2,-3,4,5,6,7,8,-9,10,11,-12});

    // Or use just a boolean. Whatever you prefer.
    net::Tensor<float> b({1,4}, true); b.fill({1,2,3,4});
    net::Tensor<float> I({2,4}, false); I.fill(1);

    x = net::function::linear(x,w,b);
    x = net::function::relu(x);
    x.perform();
    x.backward(I);

    std::cout << x << std::endl;
    std::cout << w.gradient() << std::endl;
    std::cout << b.gradient() << std::endl;
    return 0;
}
```

It produced the same results as the equivalent PyTorch code. Why am I excited? Because you can now change the internal data of the x tensor and redo all the calculations without a single memory allocation. This could be really useful for making inferences on a server because now your functions only have to change some inputs and call the perform() method.

You can also will be able to build layers like this: 
With the new object oriented interface you will be able to create models like this:

```cpp

struct Autoencoder : public net::Model<Autoencoder> {

    Autoencoder() {
        encoder.configure_optimizer(encoder_optimizer);
        decoder.configure_optimizer(decoder_optimizer);
    }

    net::layer::Sequence encoder {
        net::layer::Linear(784, 128, net::initializer::He),
        net::layer::ReLU(),
        net::layer::Linear(128, 64, net::initializer::He),
    };

    net::layer::Sequence decoder {
        net::layer::Linear(64, 128, net::initializer::He),
        net::layer::ReLU(),
        net::layer::Linear(128, 784, net::initializer::He),
        net::layer::LogSoftmax(/*axis*/ 1)
    };

    net::Tensor<float> forward(net::Tensor<float> x) {
        x = encoder(x);
        x = decoder(x);
        return x;
    }
    
    /* you can add diferent optimizers to different layers
    or the same optimizer to all of them, 
    doesn't matter since the optimizer has a shared pointer
    to it's implementation so you can pass instances of it with value
    semantics without making deep copies */

    net::optimizer::SGD encoder_optimizer {/*learning rate*/ 0.1};
    net::optimizer::SGD decoder_optimizer {/*learning rate*/ 0.2};
};

```

I used Eigen::Map for performing all operations in place without making a single copy of the data, making them highly optimized. The code is also backend-agnostic, meaning you can write your custom CUDA implementations if needed.

If you want to learn c++ or about automatic differentiation, feel free to contribute! There is a lot of work to do.

## To build the project

Please see the [contributing](.github/CONTRIBUTING.md#building-the-library) guide for more information.

## Acknowledgements

This project is being possible thanks to:

* @prince-chrismc.
