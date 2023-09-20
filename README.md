 deep-learning-library


The code isn't fully documented yet, but it's very readable and structured in a highly decoupled manner, so it can grow indefinitely. The core part, which includes things related to the internal representation of tensors, the computational graph, and design, is now functional.

The API is currently inspired by PyTorch, with one notable difference: when you perform an operation, the program doesn't actually execute it immediately. Instead, it allocates a node into a graph, waiting for you to call the perform() method on the result ( like tensorflow but this is a dynamic graph ).Here's an example I created to test it (example.cpp in the repository):

```
net::Tensor x({2,2}, { -1, 2, 5, 1 } , false);
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


To build the project:

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
g++ example.cpp -LCabernet/lib -lCabernet -I Cabernet/include
