# CaberNet C++ library

## Join the Discord:

https://discord.gg/4guRKm5x


## To do tasks:
### Short Term
- [ ] Add an optional flag to linear function, we will need to be able to do linear(x,W) without the bias parameter. This can be made simply by creating a new linear(x,W) function, and checking for
   nullptr in the internal implementation of the function.
- [ ] Add backward methods to the softmax and log_softmax functions. They are not implemented yet.
- [ ] Overload << operators for tensors so we don't have to iterate over the tensors values for print them.
- [ ] Adding more tasks here is another task!

### Long Term
- [ ] We well need to design a mechanisms for loading data into tensors. This can be challenging, since the de data lives in the tensor as a single chunk of memory,
and if we want to shuffle the batches while iterating over the Dataset for performing the SGD algorithm, the iterators of the dataset should be able to flush 
the rows of the tensors.

- [ ] We will need to add strides on the tensor data structures, for some operations.
- [ ] I will be designing the OOP interface for creating the layers and the statistical distributions modules for initializing the weights, so don't worry about them,
meanwhile there are a few files in the for you to see how the design is going.

### Considerations
If everything goes well, we can use sentenpice tokenizer creating some NLP server.

## Important

If you want to add custom functions read the README inside the src foder.


The code isn't fully documented yet, but it's very readable and structured in a highly decoupled manner, so it can grow indefinitely. The core part, which includes things related to the internal representation of tensors, the computational graph, and design, is now functional.

The API is currently inspired by PyTorch, with one notable difference: when you perform an operation, the program doesn't actually execute it immediately. Instead, it allocates a node into a graph, waiting for you to call the perform() method on the result ( like tensorflow but this is a dynamic graph ).Here's an example I created to test it (example.cpp in the repository):

## Description

```
net::Tensor x({2,2}, { -1, 2, 5, 1 } , false); // We will change this later with a fill method for loading the data into tensors, this is a very simple task to do. 
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
for (auto element : b.gradient()) std::cout << element << std::end
```

It produced the same results as the equivalent PyTorch code. Why am I excited? Because you can now change the internal data of the x tensor and redo all the calculations without a single memory allocation. This could be really useful for making inferences on a server because now your functions only have to change some inputs and call the perform() method.

I used Eigen::Map for performing all operations in place without making a single copy of the data, making them highly optimized. The code is also backend-agnostic, meaning you can write your custom CUDA implementations if needed.

If you want to learn c++ or about automatic differentiation, feel free to contribute! There is a lot of work to do.


## To build the project:

```
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
