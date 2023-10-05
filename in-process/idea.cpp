#include <iostream>
#include <vector>

namespace internal {

class Base {};

template<typename T>


class  {};

class Tensor : public Array<float> {  
};

}

namespace net {

class float_32;

template<typename Type = float_32>
class Tensor {
    public:
    using size_type = std::size_t;
    using shape_type = std::vector<size_type>;

    Tensor& self() { return *static_cast<Tensor*>(this); }
    const Tensor& self() const { return *static_cast<const Tensor*>(this); }
};

class float_32 : public Tensor<float_32> {

};

class integer_8 : public Tensor<integer_8> {

};

Tensor<float_32> fn(Tensor<float_32> x) {
    return x;
}

} // namespace net

int main() {
    Tensor x;
    Tensor y = fn(x);
}