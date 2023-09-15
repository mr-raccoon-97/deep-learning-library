#include <iostream>
#include <memory>

struct Tensor {
    Tensor(float data) : data(std::make_shared<float>(data)) {}
    std::shared_ptr<float> data;
    void print() { std::cout << *data << std::endl; }
};

int main() {
    Tensor t1(1.0);
    Tensor t2(2.0);
    Tensor t3(3.0);
    
    t2 = t1;
    t1 = t3;

    t1.print();

    return 0;
}