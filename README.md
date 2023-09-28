# CaberNet C++ Deep Learning Library

## Join the Discord:

https://discord.gg/5Z7Xfs8Q

## Description

If you want to contribute but don't understand my code, don't be afraid to ask, write in the discord server or send me an email : eric.m.cardozo@gmail.com

The code isn't fully documented yet, but it's very readable and structured in a highly decoupled manner, so it can grow indefinitely. The core part, which includes things related to the internal representation of tensors, the computational graph, and design, is now functional.

The API is currently inspired by PyTorch, with one notable difference: when you perform an operation, the program doesn't actually execute it immediately. Instead, it allocates a node into a graph, waiting for you to call the perform() method on the result ( like tensorflow but this is a dynamic graph ).Here's an example I created to test it:

```cpp
#include <iostream>
#include <CaberNet/CaberNet.h>

int main() {
    net::Tensor x({2,3}, false); x.fill({1,2,3,4,5,6});
    net::Tensor w({4,3}, true); w.fill({1,2,-3,4,5,6,7,8,-9,10,11,-12});
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
```

It produced the same results as the equivalent PyTorch code. Why am I excited? Because you can now change the internal data of the x tensor and redo all the calculations without a single memory allocation. This could be really useful for making inferences on a server because now your functions only have to change some inputs and call the perform() method.

You can also will be able to build layers like this: 
With the new object oriented interface you will be able to create models like this:

```cpp
#include <CaberNet/CaberNet.h>
struct Encoder : public net::base::Model {
    net::layer::Sequence layers;
    Encoder() {
        layers = {
            new net::layer::Linear(64, 32),
            new net::layer::ReLU(),
            new net::layer::Linear(32, 16),
            new net::layer::ReLU(),
        };
    }

    net::Tensor forward(net::Tensor x) override {
        return layers.forward(x);
    }
};

struct Decoder : public net::base::Model {
    net::layer::Sequence layers;
    Decoder() {
        layers = {
            new net::layer::Linear(16, 32),
            new net::layer::ReLU(),
            new net::layer::Linear(32, 64)
        };
    }

    net::Tensor forward(net::Tensor x) override {
        x = layers.forward(x);
        x = net::function::softmax(x, 1);
        return x;
    }
};

struct Autoencoder : public net::base::Model {
    Encoder encoder;
    Decoder decoder;

    Autoencoder() : encoder(), decoder() {}
    net::Tensor forward(net::Tensor x) override {
        x = encoder.forward(x);
        x = decoder.forward(x);
        return x;
    }
};
```

I used Eigen::Map for performing all operations in place without making a single copy of the data, making them highly optimized. The code is also backend-agnostic, meaning you can write your custom CUDA implementations if needed.

If you want to learn c++ or about automatic differentiation, feel free to contribute! There is a lot of work to do.


## To build the project:

```sh
mkdir build
cd build
cmake ..
make
sudo make install
```

Eigen supports cmake, so if sombody wants to integrate that support in the CMakeLists.txt would be nice. 

Be sure Eigen is installed in your system. 
Don't forget to add the path of the installed library when compiling, for example, if you want to compile example.cpp with g++:

```g++ example.cpp -LCaberNet/lib -lCaberNet -I CaberNet/include```

## Acknowledgements
This project is being possible thanks to:
* @prince-chrismc.
